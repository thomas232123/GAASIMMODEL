import numpy as np
from collections import Counter
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

# Elo-style ratings (not ranks!)
power_ratings = {
    'Dublin': 1720,
    'Kerry': 1747,
    'Donegal': 1735,
    'Galway': 1722,
    'Armagh': 1739,
    'Tyrone': 1625,
    'Monaghan': 1605,
    'Meath': 1616,
    'Down': 1568,
    'Cavan': 1550,
    'Louth': 1574,
    'Cork': 1564
}

prelim_matches = [
    ('Dublin', 'Cork'),
    ('Kerry', 'Cavan'),
    ('Down', 'Galway'),
    ('Donegal', 'Louth')
]

static_teams = ['Meath', 'Tyrone', 'Monaghan', 'Armagh']

# Store simulation results globally
stored_results = None
stored_stats = None

# Win probability function
def win_prob(a, b):
    diff = power_ratings[a] - power_ratings[b]
    return 1 / (1 + 10**(-diff / 400))

def simulate_score(team_a, team_b):
    """Simulate a realistic GAA score for team_a vs team_b."""
    base_goals = 1
    base_points = 10

    rating_a = power_ratings[team_a]
    rating_b = power_ratings[team_b]
    rating_diff = rating_a - rating_b

    # Bias for goals and points based on rating difference
    # Stronger team likely to score more goals/points
    mean_goals_a = base_goals + max(0, rating_diff) / 200
    mean_goals_b = base_goals + max(0, -rating_diff) / 200
    mean_points_a = base_points + max(0, rating_diff) / 50
    mean_points_b = base_points + max(0, -rating_diff) / 50

    # Simulate goals and points using Poisson distribution for realism
    goals_a = np.random.poisson(mean_goals_a)
    goals_b = np.random.poisson(mean_goals_b)
    points_a = np.random.poisson(mean_points_a)
    points_b = np.random.poisson(mean_points_b)

    # Calculate totals to decide winner
    total_a = goals_a * 3 + points_a
    total_b = goals_b * 3 + points_b

    if total_a == total_b:
        # Handle draw by adding a random extra point to a random team
        if np.random.rand() < 0.5:
            points_a += 1
            total_a += 1
        else:
            points_b += 1
            total_b += 1

    winner = team_a if total_a > total_b else team_b

    score_a = f"{goals_a}-{points_a}"
    score_b = f"{goals_b}-{points_b}"

    return winner, score_a, score_b

# Simulate match outcome
def simulate_match(a, b):
    winner, score_a, score_b = simulate_score(a, b)
    return winner, score_a, score_b

# Run one simulation and return full path
def simulate_one_full_path():
    sim_path = {"Prelim": [], "QF": [], "SF": [], "Final": ""}

    # For tracking teams per round
    teams_in_prelim = set()
    teams_in_qf = set()
    teams_in_sf = set()
    teams_in_final = set()

    # Prelim QFs - fixed matches as in real championship
    prelim_winners = []
    for a, b in prelim_matches:
        teams_in_prelim.update([a, b])
        w, score_a, score_b = simulate_match(a, b)
        prelim_winners.append(w)
        sim_path["Prelim"].append((a, b, w, score_a, score_b))

    # Quarter-Finals - draw between prelim winners and seeded teams
    seeded_teams = static_teams.copy()  # ['Meath', 'Tyrone', 'Monaghan', 'Armagh']
    np.random.shuffle(prelim_winners)
    np.random.shuffle(seeded_teams)

    qf_winners = []
    for i in range(4):
        # Each QF is seeded team vs prelim winner
        a, b = seeded_teams[i], prelim_winners[i]
        teams_in_qf.update([a, b])
        w, score_a, score_b = simulate_match(a, b)
        qf_winners.append(w)
        sim_path["QF"].append((a, b, w, score_a, score_b))

    # Semi-Finals - random draw from QF winners
    teams_in_sf.update(qf_winners)
    np.random.shuffle(qf_winners)

    sf1, sf1_score_a, sf1_score_b = simulate_match(qf_winners[0], qf_winners[1])
    sf2, sf2_score_a, sf2_score_b = simulate_match(qf_winners[2], qf_winners[3])
    sim_path["SF"].append((qf_winners[0], qf_winners[1], sf1, sf1_score_a, sf1_score_b))
    sim_path["SF"].append((qf_winners[2], qf_winners[3], sf2, sf2_score_a, sf2_score_b))

    # Final
    teams_in_final.update([sf1, sf2])
    champ, final_score_a, final_score_b = simulate_match(sf1, sf2)
    sim_path["Final"] = (sf1, sf2, champ, final_score_a, final_score_b)

    return champ, sim_path, teams_in_prelim, teams_in_qf, teams_in_sf, teams_in_final

# Run all simulations and store paths
def run_all_sims(n=10000):
    results = []

    # Counters to track round appearances
    prelim_counts = Counter()
    qf_counts = Counter()
    sf_counts = Counter()
    final_counts = Counter()
    win_counts = Counter()

    for _ in range(n):
        champ, path, pre, qf, sf, final = simulate_one_full_path()
        results.append((champ, path))
        for t in pre:
            prelim_counts[t] += 1
        for t in qf:
            qf_counts[t] += 1
        for t in sf:
            sf_counts[t] += 1
        for t in final:
            final_counts[t] += 1
        win_counts[champ] += 1

    return results, prelim_counts, qf_counts, sf_counts, final_counts, win_counts

@app.route('/')
def index():
    global stored_results, stored_stats

    # Run simulations if not already done
    if stored_results is None:
        stored_results, prelim, qf, sf, final, wins = run_all_sims(10000)
        stored_stats = {
            'wins': wins,
            'prelim': prelim,
            'qf': qf,
            'sf': sf,
            'final': final
        }

    # Prepare win percentages
    win_percentages = []
    for team, count in sorted(stored_stats['wins'].items(), key=lambda x: -x[1]):
        win_percentages.append({
            'team': team,
            'percentage': round(count / 100, 2)
        })

    return render_template('index.html', 
                         win_percentages=win_percentages,
                         teams=list(power_ratings.keys()))

@app.route('/simulation/<int:sim_id>')
def view_simulation(sim_id):
    global stored_results

    if stored_results is None or sim_id < 1 or sim_id > len(stored_results):
        return jsonify({'error': 'Invalid simulation ID'}), 400

    champ, path = stored_results[sim_id - 1]
    return jsonify({
        'simulation_id': sim_id,
        'champion': champ,
        'path': path
    })

@app.route('/team_wins/<team>')
def team_wins(team):
    global stored_results

    if stored_results is None:
        return jsonify({'error': 'No simulations found'}), 400

    team_lower = team.lower()
    matching_sims = []

    for i, (champ, _) in enumerate(stored_results):
        if champ.lower() == team_lower:
            matching_sims.append(i + 1)

    return jsonify({
        'team': team,
        'total_wins': len(matching_sims),
        'simulation_ids': matching_sims[:100]  # Limit to first 100
    })

@app.route('/round_stats')
def round_stats():
    global stored_stats

    if stored_stats is None:
        return jsonify({'error': 'No statistics found'}), 400

    def format_round_stats(counts, round_name):
        stats = []
        # Show all teams, including those with 0 appearances
        for team in power_ratings.keys():
            count = counts.get(team, 0)
            stats.append({
                'team': team,
                'appearances': count,
                'percentage': round(count / 100, 2)
            })
        # Sort by percentage descending
        stats.sort(key=lambda x: -x['percentage'])
        return stats

    return jsonify({
        'prelim': format_round_stats(stored_stats['prelim'], 'Preliminary'),
        'qf': format_round_stats(stored_stats['qf'], 'Quarter-Finals'),
        'sf': format_round_stats(stored_stats['sf'], 'Semi-Finals'),
        'final': format_round_stats(stored_stats['final'], 'Finals')
    })

@app.route('/rerun_simulations', methods=['POST'])
def rerun_simulations():
    global stored_results, stored_stats

    # Clear existing results and re-run simulations
    stored_results, prelim, qf, sf, final, wins = run_all_sims(10000)
    stored_stats = {
        'wins': wins,
        'prelim': prelim,
        'qf': qf,
        'sf': sf,
        'final': final
    }

    # Prepare fresh win percentages
    win_percentages = []
    for team, count in sorted(stored_stats['wins'].items(), key=lambda x: -x[1]):
        win_percentages.append({
            'team': team,
            'percentage': round(count / 100, 2)
        })

    return jsonify({
        'success': True,
        'message': 'New simulations completed!',
        'win_percentages': win_percentages
    })

@app.route('/head_to_head/<team1>/<team2>')
def head_to_head(team1, team2):
    global stored_results

    if stored_results is None:
        return jsonify({'error': 'No simulations found'}), 400

    if team1 not in power_ratings or team2 not in power_ratings:
        return jsonify({'error': 'Invalid team names'}), 400

    matchups = []
    team1_wins = 0
    team2_wins = 0

    # Search through all simulations for matches between these teams
    for sim_id, (champ, path) in enumerate(stored_results, 1):
        for round_name, round_matches in path.items():
            if round_name == "Final":
                # Final is stored differently (single match tuple)
                if isinstance(round_matches, tuple):
                    team_a, team_b, winner, score_a, score_b = round_matches
                    if (team_a == team1 and team_b == team2) or (team_a == team2 and team_b == team1):
                        matchups.append({
                            'simulation_id': sim_id,
                            'round': round_name,
                            'team_a': team_a,
                            'team_b': team_b,
                            'score_a': score_a,
                            'score_b': score_b,
                            'winner': winner
                        })
                        if winner == team1:
                            team1_wins += 1
                        else:
                            team2_wins += 1
            else:
                # Other rounds are lists of matches
                for match in round_matches:
                    team_a, team_b, winner, score_a, score_b = match
                    if (team_a == team1 and team_b == team2) or (team_a == team2 and team_b == team1):
                        matchups.append({
                            'simulation_id': sim_id,
                            'round': round_name,
                            'team_a': team_a,
                            'team_b': team_b,
                            'score_a': score_a,
                            'score_b': score_b,
                            'winner': winner
                        })
                        if winner == team1:
                            team1_wins += 1
                        else:
                            team2_wins += 1

    # Calculate win percentages
    total_matches = len(matchups)
    team1_win_rate = round((team1_wins / total_matches * 100), 1) if total_matches > 0 else 0
    team2_win_rate = round((team2_wins / total_matches * 100), 1) if total_matches > 0 else 0

    # Get expected win probability based on ratings
    expected_team1_prob = round(win_prob(team1, team2) * 100, 1)
    expected_team2_prob = round(100 - expected_team1_prob, 1)

    return jsonify({
        'team1': team1,
        'team2': team2,
        'total_matches': total_matches,
        'team1_wins': team1_wins,
        'team2_wins': team2_wins,
        'team1_win_rate': team1_win_rate,
        'team2_win_rate': team2_win_rate,
        'expected_team1_prob': expected_team1_prob,
        'expected_team2_prob': expected_team2_prob,
        'matchups': matchups[:50]  # Limit to first 50 for performance
    })

@app.route('/team_summary/<team>')
def team_summary(team):
    global stored_stats, stored_results

    if stored_stats is None or stored_results is None:
        return jsonify({'error': 'No statistics found'}), 400

    # Get team's performance across all rounds
    prelim_rate = round(stored_stats['prelim'].get(team, 0) / 100, 2)
    qf_rate = round(stored_stats['qf'].get(team, 0) / 100, 2)
    sf_rate = round(stored_stats['sf'].get(team, 0) / 100, 2)
    final_rate = round(stored_stats['final'].get(team, 0) / 100, 2)
    win_rate = round(stored_stats['wins'].get(team, 0) / 100, 2)

    # Calculate team ranking based on power rating
    sorted_teams = sorted(power_ratings.items(), key=lambda x: -x[1])
    team_rank = next(i for i, (t, _) in enumerate(sorted_teams, 1) if t == team)

    # Find team's best and worst matchups based on rating differences
    best_matchups = []
    worst_matchups = []
    team_rating = power_ratings[team]

    for other_team, other_rating in power_ratings.items():
        if other_team != team:
            win_prob_against = round(1 / (1 + 10**((other_rating - team_rating) / 400)) * 100, 1)
            if win_prob_against >= 70:
                best_matchups.append((other_team, win_prob_against))
            elif win_prob_against <= 30:
                worst_matchups.append((other_team, win_prob_against))

    best_matchups.sort(key=lambda x: -x[1])
    worst_matchups.sort(key=lambda x: x[1])

    # Determine team's championship path difficulty
    if win_rate >= 20:
        tier = "Championship Contender"
    elif win_rate >= 10:
        tier = "Strong Contender"
    elif win_rate >= 5:
        tier = "Dark Horse"
    elif win_rate >= 1:
        tier = "Outside Chance"
    else:
        tier = "Long Shot"

    return jsonify({
        'team': team,
        'power_rating': power_ratings[team],
        'ranking': f"{team_rank} of {len(power_ratings)}",
        'tier': tier,
        'round_performance': {
            'prelim': prelim_rate,
            'quarter_final': qf_rate,
            'semi_final': sf_rate,
            'final': final_rate,
            'championship': win_rate
        },
        'best_matchups': best_matchups[:3],
        'worst_matchups': worst_matchups[:3],
        'total_wins': stored_stats['wins'].get(team, 0)
    })

@app.route('/bracket_visualization/<int:sim_id>')
def bracket_visualization(sim_id):
    global stored_results

    if stored_results is None or sim_id < 1 or sim_id > len(stored_results):
        return jsonify({'error': 'Invalid simulation ID'}), 400

    champ, path = stored_results[sim_id - 1]
    
    # Structure the bracket data for visualization
    bracket = {
        'prelim': [],
        'quarter_finals': [],
        'semi_finals': [],
        'final': None,
        'champion': champ
    }

    # Process preliminary matches
    for match in path['Prelim']:
        team_a, team_b, winner, score_a, score_b = match
        bracket['prelim'].append({
            'team_a': team_a,
            'team_b': team_b,
            'winner': winner,
            'score_a': score_a,
            'score_b': score_b,
            'upset': is_upset(team_a, team_b, winner)
        })

    # Process quarter-finals
    for match in path['QF']:
        team_a, team_b, winner, score_a, score_b = match
        bracket['quarter_finals'].append({
            'team_a': team_a,
            'team_b': team_b,
            'winner': winner,
            'score_a': score_a,
            'score_b': score_b,
            'upset': is_upset(team_a, team_b, winner)
        })

    # Process semi-finals
    for match in path['SF']:
        team_a, team_b, winner, score_a, score_b = match
        bracket['semi_finals'].append({
            'team_a': team_a,
            'team_b': team_b,
            'winner': winner,
            'score_a': score_a,
            'score_b': score_b,
            'upset': is_upset(team_a, team_b, winner)
        })

    # Process final
    team_a, team_b, winner, score_a, score_b = path['Final']
    bracket['final'] = {
        'team_a': team_a,
        'team_b': team_b,
        'winner': winner,
        'score_a': score_a,
        'score_b': score_b,
        'upset': is_upset(team_a, team_b, winner)
    }

    return jsonify(bracket)



def is_upset(team_a, team_b, winner):
    """Determine if a match result is an upset based on power ratings (50+ point gap)"""
    rating_a = power_ratings[team_a]
    rating_b = power_ratings[team_b]
    
    # Only consider it an upset if rating gap is 50+ points
    if winner == team_a and rating_a < rating_b and (rating_b - rating_a) >= 50:
        return True
    elif winner == team_b and rating_b < rating_a and (rating_a - rating_b) >= 50:
        return True
    return False

def calculate_upset_magnitude(team_a, team_b, winner):
    """Calculate the magnitude of an upset (rating difference)"""
    rating_a = power_ratings[team_a]
    rating_b = power_ratings[team_b]
    
    if winner == team_a and rating_a < rating_b:
        return rating_b - rating_a
    elif winner == team_b and rating_b < rating_a:
        return rating_a - rating_b
    return 0



if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
