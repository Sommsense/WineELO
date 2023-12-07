import pandas as pd
import datetime
from dateutil.relativedelta import relativedelta
import numpy as np
from itertools import combinations
pd.set_option('mode.chained_assignment', None)
from .chessratings import uscf_elo
import numpy as np
from datetime import date
from collections import Counter

'''
This package is an implementation of the US Chess Federation (USCF) rating system for Regular, Over-The-Board (OTB) events. Details can be found here: http://www.glicko.net/ratings/rating.system.pdf

When initializing ratings for unrated players, we use the recommended age-based initialization logic. If no age is provided, we assume that a player is an adult above the age of 26.

This package only implements partial logic for rating floors, disregarding rating floor logic for players with an original Life Master (OLM) title or the effect of large cash prizes.

At this time, this package does not allow for updating USCF ratings based on results from foreign FIDE events.
'''


# universal settings for the rating algorithm
epsilon_special_rating = 10**-7
absolute_rating_floor = 100
B = 14


# A player is a participant in a tournament and needs to be initialized
class Player:

    def __init__(self, id, rating, nr_games_played=100, nr_wins=50, nr_losses=50, nr_tournaments=0, birth_date=date(1990, 1, 1), current_date=date(2021, 1, 1), Nr=0):
        self.id = id
        self.nr_games_played = nr_games_played
        self.nr_wins = nr_wins
        self.nr_draws = nr_games_played - nr_wins - nr_losses
        self.nr_losses = nr_losses
        self.rating = rating
        self._nr_tournaments = nr_tournaments
        self.birth_date = birth_date
        self.current_date = current_date
        # Nr is the number of events (tournaments) in which a player completed three rated games
        self.Nr = Nr
        self.initialized_rating = self.initialize_rating()
        self.estimated_rating = self.initialized_rating
        self.established_rating = self.determine_established_rating()
        self.effective_nr_games = self.compute_effective_nr_games()
        self.rating_type = self.compute_rating_type()

    # players with > 25 games are deemed to have an 'established' rating
    def determine_established_rating(self):
        if self.nr_games_played > 25:
            established_rating = True
        else:
            established_rating = False
        return established_rating

    # age-based rating initialization for players that are unrated
    def _compute_age_based_rating(self):
        age = (self.current_date - self.birth_date).days/365.25
        if age < 2:
            rating = 100
        elif 2 <= age <= 26:
            rating = 50*age
        else:
            rating = 1300
        return rating

    # initialize the rating - based on age for unrated players, and based on the actual rating for rated players
    def initialize_rating(self):
        if self.rating is None:
            initialized_rating = self._compute_age_based_rating()
        else:
            initialized_rating = self.rating
        return initialized_rating

    # computation of a quantity called the effective number of games, used to convey the approximate reliability of a rating on the scale of a game count
    def compute_effective_nr_games(self):
        if self.initialized_rating <= 2355:
            n = 50/np.sqrt(0.662 + 0.00000739*(2569 -
                           self.initialized_rating)**2)
        else:
            n = 50

        effective_nr_games = min(n, self.nr_games_played)
        return effective_nr_games

    # categorization of players to determine whether the special or standard rating algorithm should be used
    def compute_rating_type(self):
        if self.nr_games_played <= 8:
            rating_type = 'special-new'
        elif self.nr_wins == self.nr_games_played:
            rating_type = 'special-only-wins'
        elif self.nr_losses == self.nr_games_played:
            rating_type = 'special-only-losses'
        else:
            rating_type = 'standard'
        return rating_type


# A tournament is initialized with a list of players
class Tournament:

    # Note: tournament_results needs to be in a specific format:
    # [ [ ( player A ID, player B ID ), winner ID ], [ ( player B ID , player C ID ), winner ID ], ... ]
    # winner is the ID of the player who won. If a draw, the value is null.

    def __init__(self, players, tournament_results, tournament_date=date.today(), time_control_minutes=60, time_control_increment_seconds=0):

        self.nr_games_tournament = len(tournament_results)
        self.tournament_results = tournament_results
        self.tournament_date = tournament_date
        self.time_control_minutes = time_control_minutes
        self.time_control_increment_seconds = time_control_increment_seconds
        self.players = players
        self.individual_match = self._verify_individual_match()
        self.valid_tournament = self._valid_tournament()
        self.nr_players = len(self.players)
        self._compute_estimated_ratings()

    def __iter___(self):
        return iter(self.players)

    # only tournaments that satisfy certain criteria are valid
    def _valid_tournament(self):
        if len(self.players) <= 1:
            return False
        elif self.individual_match:
            player_1 = self.players[0]
            player_2 = self.players[1]
            if player_1.established_rating is False or player_2.established_rating is False or abs(player_1.initialized_rating - player_2.initialized_rating) > 400:
                return False
            else:
                return True
        else:
            return True

    # an individual match is defined as a matchup between exactly two players
    def _verify_individual_match(self):
        if len(self.players) == 2:
            return True
        else:
            return False

    # estimated rating is an modification of the initial rating, only applied to players that do not have any matches to their name yet
    def _compute_estimated_ratings(self):
        for p in self.players:
            if p.nr_games_played == 0:
                nr_matches = 0
                score = 0
                opponent_ratings = []
                for m in self.tournament_results:
                    pairing = m[0]

                    if p.id in pairing:
                        opponent_id = [o for o in pairing if o != p.id][0]
                        opponent = [
                            p for p in self.players if p.id == opponent_id][0]
                        opponent_rating = opponent.initialized_rating
                        opponent_ratings.append(opponent_rating)

                        nr_matches += 1

                        if p.id == m[1]:
                            score += 1
                        elif opponent_id != m[1]:
                            score += 0.5

                p.estimated_rating = max(100, (p.initialized_rating + sum(
                    opponent_ratings) + 400 * (2*score - nr_matches))/(1 + nr_matches))

    # key function that runs a tournament by testing whether it is valid, initializing PlayerTournament classes and computing all rating updates.
    # outputs a list of lists with the following information:
    # [ [ player_1_id, tournament_date, player_1_tournament_number, tournament_matches_played, wins, draws, losses, updated_rating ], ... ]
    def run(self):
        if self.valid_tournament:
            updated_info = []
            for p in self.players:

                playertournament = _PlayerTournament(p, self)
                new_rating = playertournament.update_player_rating()
                player_tournament_info = [p.id, self.tournament_date, playertournament._tournament_nr, playertournament.nr_matches,
                                          playertournament.nr_wins, playertournament.nr_draws, playertournament.nr_losses, new_rating]

                p.rating = new_rating
                updated_info.append(player_tournament_info)

            return updated_info
        else:
            return None


# PlayerTournament is a hidden class that is used for methods that relate to an individual player's participation in a tournament
class _PlayerTournament:

    def __init__(self, player, tournament):
        self.player = player
        self.tournament = tournament
        self._matches = self._player_matches()
        self.score = self._tournament_score()
        self.nr_matches, self.nr_wins, self.nr_draws, self.nr_losses = self._tournament_stats()
        self._tournament_nr = self.player._nr_tournaments + 1
        self._opponents = self._retrieve_opponents()
        self._opponent_ratings = [
            o.estimated_rating for o in self._opponents]
        self._adjusted_initialized_rating, self._adjusted_score = self._compute_adjusted_initialized_rating_and_score()

    def __iter___(self):
        return iter(self._opponents)

    # retrieve all the tournament matches that a player participated in
    def _player_matches(self):
        player_matches = []
        for m in self.tournament.tournament_results:
            if self.player.id in m[0]:
                player_matches.append(m)
        return player_matches

    # retrieve the Player objects for all the opponents a player faces in a tournament
    def _retrieve_opponents(self):
        opponents = []
        for m in self._matches:
            pairing = m[0]
            if self.player.id in pairing:
                opponent_id = [o for o in pairing if o != self.player.id][0]
                opponent = [
                    p for p in self.tournament.players if p.id == opponent_id][0]
                opponents.append(opponent)

        return opponents

    # compute the performance of a player in a given match (win = 1, draw = 0.5 and loss = 0 points)
    def _compute_match_performance(self, player_id, match):
        if player_id == match[1]:
            score = 1
        elif match[1] in [p.id for p in self.tournament.players]:
            score = 0
        else:
            score = 0.5

        return score

    # compute the total tournament score (sum of match performances) for a given player
    def _tournament_score(self):
        tournament_score = 0
        for m in self._matches:
            match_score = self._compute_match_performance(self.player.id, m)
            tournament_score += match_score

        return tournament_score

    # compute the total number of games, wins, draws and losses for a player in a tournament
    def _tournament_stats(self):
        nr_games = len(self._matches)
        nr_wins = 0
        nr_draws = 0
        nr_losses = 0
        for t in self._matches:
            score = self._compute_match_performance(self.player.id, t)
            if score == 0:
                nr_losses += 1
            elif score == 0.5:
                nr_draws += 1
            else:
                nr_wins += 1

        return nr_games, nr_wins, nr_draws, nr_losses

   # an adjusted initialized rating and adjusted score are used as variables to calculate the special rating
    def _compute_adjusted_initialized_rating_and_score(self):

        # players with <= 8 games, or players that have had only wins/losses in all previous rated games, get a special rating
        if self.player.rating_type == 'special-only-wins':
            adjusted_initialized_rating = self.player.estimated_rating - 400
            adjusted_score = self.score + self.player.effective_nr_games
        elif self.player.rating_type == 'special-only-losses':
            adjusted_initialized_rating = self.player.estimated_rating + 400
            adjusted_score = self.score
        else:
            adjusted_initialized_rating = self.player.estimated_rating
            adjusted_score = self.score + self.player.effective_nr_games/2

        return adjusted_initialized_rating, adjusted_score

    # provisional winning expectancy, used to estimate the likelihood of a victory for a player rated with the special rating algorithm
    def _compute_provisional_winning_expectancy(self, player_rating, opponent_rating):

        if player_rating <= opponent_rating - 400:
            pwe = 0
        elif opponent_rating - 400 < player_rating < opponent_rating + 400:
            pwe = 0.5 + (player_rating - opponent_rating)/800
        else:
            pwe = 1
        return pwe

    # an objective function for the special rating. The goal is to determine the value of the special rating estimate (R) such that the value of this objecive is equal to 0 within reasonable tolerance
    def _special_rating_objective(self, special_rating_estimate):

        pwe_factor = self._compute_provisional_winning_expectancy(
            special_rating_estimate, self._adjusted_initialized_rating)

        sum_pwe = sum([self._compute_provisional_winning_expectancy(
            special_rating_estimate, o.estimated_rating) for o in self._opponents])

        objective_fn = self.player.effective_nr_games * \
            pwe_factor + sum_pwe - self._adjusted_score

        return objective_fn

    # Sz is used in the iterative procedure to find the solution for the special rating objective function
    def _compute_Sz(self):
        Sz = [o + 400 for o in self._opponent_ratings] + \
            [o - 400 for o in self._opponent_ratings] + \
            [self._adjusted_initialized_rating - 400] + \
            [self._adjusted_initialized_rating + 400]
        return Sz

    # the first step of the iterative algorithm - provides the first estimate of the special rating
    def _special_rating_step_1_compute_M(self):

        M = (self.player.effective_nr_games*self._adjusted_initialized_rating + sum(self._opponent_ratings) +
             400 * (2*self.score - self.nr_matches))/(self.player.effective_nr_games + self.nr_matches)
        return M

    # the second step of the iterative process to find the special rating
    def _special_rating_step_2(self, M, f_M, Sz):
        step_2_satisfied = False
        while step_2_satisfied is False:
            if f_M > epsilon_special_rating:
                # Let za be the largest value in Sz for which M > za.

                za = max([z for z in Sz if z < M])
                f_za = self._special_rating_objective(za)

                if abs(f_M - f_za) < epsilon_special_rating:
                    M = za
                    f_M = f_za
                    continue
                else:
                    M_star = M - f_M * ((M - za) / (f_M - f_za))
                    if M_star < za:
                        M = za
                        f_M = f_za
                        continue
                    elif za <= M_star < M:
                        M = M_star
                        f_M = self._special_rating_objective(M_star)
                        continue
                    else:
                        step_2_satisfied = True
                        break
            else:
                step_2_satisfied = True
                break

        return M, f_M

    # the third step of the iterative process to find the special rating
    def _special_rating_step_3(self, M, f_M, Sz):
        step_3_satisfied = False
        while step_3_satisfied is False:
            if f_M < -epsilon_special_rating:

                zb = min([z for z in Sz if z > M])
                f_zb = self._special_rating_objective(zb)
                if abs(f_zb - f_M) < epsilon_special_rating:
                    M = zb
                    f_M = f_zb
                else:
                    M_star = M - f_M * ((zb - M) / (f_zb - f_M))
                    if M_star > zb:
                        M = zb
                        f_M = self._special_rating_objective(M)

                        continue
                    elif M < M_star <= zb:
                        M = M_star
                        f_M = self._special_rating_objective(M)
                        continue
                    else:
                        step_3_satisfied = True
                        return M, f_M
            else:
                step_3_satisfied = True
                return M, f_M

    # the fourth step of the iterative process to find the special rating
    def _special_rating_step_4(self, f_M, M, Sz):
        if abs(f_M) <= epsilon_special_rating:
            p = len([o for o in self._opponent_ratings if abs(M - o) <= 400])

            if abs(M - self._adjusted_initialized_rating) <= 400:
                p += 1

            if p > 0:
                return M
            elif p == 0:
                za = max([s for s in Sz if s < M])
                zb = min([s for s in Sz if s > M])
                if za <= self.player.estimated_rating <= zb:
                    M = self.player.estimated_rating
                elif self.player.estimated_rating < za:
                    M = za
                elif self.player.estimated_rating > zb:
                    M = zb
                else:
                    raise Exception(
                        'M is outside the range of expected values.')

        return M

    # call the step 1 - 4 functions to compute the special rating
    def _compute_special_rating(self):

        M = self._special_rating_step_1_compute_M()
        f_M = self._special_rating_objective(M)
        Sz = self._compute_Sz()

        M, f_M = self._special_rating_step_2(M, f_M, Sz)

        M, f_M = self._special_rating_step_3(M, f_M, Sz)

        M = self._special_rating_step_4(f_M, M, Sz)

        M = min(2700, M)
        return M

    # the winning expectancy of a player in a given matchup, under the standard rating logic
    def _compute_standard_winning_expectancy(self, player_rating, opponent_rating):
        winning_expectancy = 1 / \
            (1+10**-((player_rating - opponent_rating)/400))
        return winning_expectancy

    # the value of K is an important component in calculating changes in the standard rating
    def _compute_standard_rating_K(self, rating, effective_nr_games, nr_tournament_matches):

        K = 800/(effective_nr_games + nr_tournament_matches)

        if 30 <= (self.tournament.time_control_minutes + self.tournament.time_control_increment_seconds) <= 65 and rating > 2200:
            if rating < 2500:
                K = (800 * (6.5 - 0.0025*rating))/(effective_nr_games +
                                                   nr_tournament_matches)
            else:
                K = 200/(effective_nr_games +
                         nr_tournament_matches)
        return K

    # the standard rating function is used for players with N > 8 who have not had either all wins or all losses in every previous rated game
    def _compute_standard_rating(self, bonus=True):
        sum_swe = sum([self._compute_standard_winning_expectancy(
            self.player.estimated_rating, o.initialized_rating) for o in self._opponents])

        K = self._compute_standard_rating_K(
            self.player.estimated_rating, self.player.effective_nr_games, self.nr_matches)

        opponent_ids = [o.id for o in self._opponents]
        max_nr_games_one_opponent = max(
            Counter(opponent_ids).values())

        if self.nr_matches < 3 or max_nr_games_one_opponent > 2:

            rating_new = self.player.estimated_rating + \
                K*(self.score - sum_swe)
        else:
            if bonus:
                rating_new = self.player.estimated_rating + K*(self.score - sum_swe) + max(
                    0, K*(self.score - sum_swe) - B*np.sqrt(max(self.nr_matches, 4)))

            else:
                rating_new = self.player.estimated_rating + \
                    K*(self.score - sum_swe)

        return rating_new

    # all ratings are subject to a floor of 100, and this may be higher depending on the conditions outlined below
    def _compute_rating_floor(self):

        # number of total wins after the tournament
        Nw = self.player.nr_wins + self.nr_wins

        # number of total draws after the tournament
        Nd = self.player.nr_draws + self.nr_draws

        # number of events in which a player has completed three rating games. defaults to 0 when class initialized, but other value can be specified
        # NOTE: can be updated
        if self.nr_matches >= 3:
            self.player.Nr += 1

        otb_absolute_rating_floor = min(
            absolute_rating_floor + 4*Nw + 2*Nd + self.player.Nr, 150)

        # a player with an established rating has a rating floor possibly higher than the absolute floor. Higher rating floors exists at 1200 - 2100
        if self.player.estimated_rating >= 1200 and self.player.established_rating is True:
            otb_absolute_rating_floor = int(
                (self.player.estimated_rating - 200) / 100)*100

        return otb_absolute_rating_floor

    # function to update the rating of a player after a tournament
    def update_player_rating(self):

        # Valid individual matches are rated differently. The bonus formula does not apply, and the maximum rating change in a match is 50 points. Rating floors do not apply.
        rating_floor = self._compute_rating_floor()
        if self.tournament.individual_match:
            updated_rating = self._compute_standard_rating(bonus=False)
            updated_rating_bounded = min(max(
                self.player.estimated_rating - 50, updated_rating), self.player.estimated_rating + 50, updated_rating)
            return updated_rating_bounded
        else:
            if self.player.rating_type == 'standard':
                updated_rating = self._compute_standard_rating()
            else:
                updated_rating = self._compute_special_rating()

            updated_rating_bounded = max(updated_rating, rating_floor)
            return updated_rating_bounded


def nearest(items, pivot):
    return min(items, key=lambda x: abs(x - pivot))


def compute_date(scrape_date, review_date, review_time_ago):
    review_month = review_date[5:8]
    review_day = review_date[-20:-18].strip()

    if 'over' in review_time_ago:
        crop_offset_string = review_time_ago.split('over')[1].strip()
        offset_period = int(crop_offset_string[:2].strip())

        min_date = scrape_date - relativedelta(months=12*(offset_period+1))
        max_date = scrape_date - relativedelta(months=12*offset_period)

        candidate_years = [min_date.year, max_date.year]
        candidate_dates = [datetime.datetime.strptime(
            review_day + ' ' + review_month + ' ' + str(y), '%d %b %Y') for y in candidate_years]

        final_review_date = [
            d for d in candidate_dates if d > min_date and d < max_date][0]

    elif 'almost' in review_time_ago:
        crop_offset_string = review_time_ago.split('almost')[1].strip()
        offset_period = int(crop_offset_string[:2].strip())

        min_date = scrape_date - relativedelta(months=12*offset_period)
        max_date = scrape_date

        candidate_years = [min_date.year, max_date.year]
        candidate_dates = [datetime.datetime.strptime(
            review_day + ' ' + review_month + ' ' + str(y), '%d %b %Y') for y in candidate_years]

        final_review_date = [
            d for d in candidate_dates if d > min_date and d < max_date][0]

    else:
        if 'about' in review_time_ago:
            crop_offset_string = review_time_ago.split('about')[1].strip()
            offset_period = int(crop_offset_string[:2].strip())
        else:
            offset_period = int(review_time_ago[:2].strip())

        if 'month' in review_time_ago:
            offset_scrape_date = scrape_date - \
                relativedelta(months=offset_period)
        elif 'year' in review_time_ago:
            offset_scrape_date = scrape_date - \
                relativedelta(months=12*offset_period)
        else:
            offset_scrape_date = scrape_date

        candidate_years = [offset_scrape_date.year - 1,
                           offset_scrape_date.year, offset_scrape_date.year + 1]
        try:
            candidate_dates = [datetime.datetime.strptime(
                review_day + ' ' + review_month + ' ' + str(y), '%d %b %Y') for y in candidate_years]
        # in some fringe cases, we may be dealing with February 29th, which only exists on leap years
        except ValueError:
            candidate_dates = [datetime.datetime.strptime(str(int(
                review_day) - 1) + ' ' + review_month + ' ' + str(y), '%d %b %Y') for y in candidate_years]

        final_review_date = nearest(candidate_dates, offset_scrape_date)

    return final_review_date


def clean_wine_reviews(review_df):

    review_df['review_date'] = review_df.apply(lambda x: compute_date(
        x['scrape_date'], x['review_date'], x['review_time_ago']), axis=1)

    # drop any reviews that don't have a vintage specified. N.V. is acceptable, but blank vintage is not.
    review_df['vintage'].replace({'': np.nan}, inplace=True)
    review_df.dropna(subset=['vintage'], axis=0, inplace=True)

    just_reviews = review_df[['wine_id',
                              'reviewer', 'rating', 'review_date']]
    return just_reviews


def compute_head_to_head_result(wine_0, wine_1, rating_0, rating_1):
    if rating_0 > rating_1:
        return wine_0
    elif rating_0 < rating_1:
        return wine_1
    else:
        return np.nan


def player_info_lookup(wine_id, score_lookup_table):
    score_lookup_table_filtered = score_lookup_table.loc[
        score_lookup_table['wine_id'] == wine_id]
    if score_lookup_table_filtered.empty:
        elo_rating = None
        tournament_number = 0
        nr_games_played = 0
        nr_wins = 0
        nr_losses = 0
    else:
        score_lookup = score_lookup_table_filtered.loc[score_lookup_table_filtered['tournament_number'] == max(
            score_lookup_table_filtered['tournament_number'])].iloc[0]
        elo_rating = score_lookup['elo_rating']
        tournament_number = score_lookup['tournament_number']
        nr_games_played = sum(score_lookup_table_filtered['nr_games_played'])
        nr_wins = sum(score_lookup_table_filtered['nr_wins'])
        nr_losses = sum(score_lookup_table_filtered['nr_losses'])

    return elo_rating, tournament_number, nr_games_played, nr_wins, nr_losses


def match_format(combo, review_table):
    id_0 = combo[0]
    id_1 = combo[1]
    rating_0 = review_table.at[combo[0], 'rating']
    rating_1 = review_table.at[combo[1], 'rating']

    # note: still need to fix this
    if isinstance(rating_0, float) and isinstance(rating_1, float):
        result = compute_head_to_head_result(id_0, id_1, rating_0, rating_1)
        match_result = ((id_0, id_1), result)

        return match_result


def run_single_tournaments(r, review_df, score_lookup_table):

    review_df_date = review_df.loc[review_df['review_date'] == r]
    reviewers = sorted(list(set(review_df_date['reviewer'])))
    for u in reviewers:
        review_df_slice = review_df_date.loc[review_df_date['reviewer'] == u]
        # In some rare cases, an individual may have rated an individual wine more than once in one day. In this case, we eliminate one of these reviews
        review_df_slice = review_df_slice[~review_df_slice.index.duplicated(
            keep='first')]

        players = []
        unique_players = list(set(review_df_slice.index))

        for up in unique_players:
            rating, tournament_number, nr_games_played, nr_wins, nr_losses = player_info_lookup(
                up, score_lookup_table)
            p = uscf_elo.Player(up, rating, nr_games_played,
                                nr_wins, nr_losses, tournament_number)
            players.append(p)

        combos = list(combinations(review_df_slice.index, 2))
        tournament_results = []
        for c in combos:
            match_result = match_format(c, review_df_slice)
            tournament_results.append(match_result)

        tournament = uscf_elo.Tournament(
            players=players, tournament_results=tournament_results, tournament_date=r)
        if tournament.valid_tournament:
            try:
                updated_scores = tournament.run()
                updated_scores_with_reviewer = [
                    v + [u] for v in updated_scores]

                score_lookup_entry_table = pd.DataFrame(updated_scores_with_reviewer, columns=[
                                                        'wine_id', 'tournament_date', 'tournament_number', 'nr_games_played', 'nr_wins', 'nr_draws', 'nr_losses', 'elo_rating', 'reviewer'])
                score_lookup_table = score_lookup_table.append(
                    score_lookup_entry_table)
            except:
                continue



def run_tournaments(review_df, score_lookup_table):
    review_dates = sorted(list(set(review_df['review_date'])))

    for r in review_dates:

        review_df_date = review_df.loc[review_df['review_date'] == r]
        reviewers = sorted(list(set(review_df_date['reviewer'])))
        for u in reviewers:
            review_df_slice = review_df_date.loc[review_df_date['reviewer'] == u]
            # In some rare cases, an individual may have rated an individual wine more than once in one day. In this case, we eliminate one of these reviews
            review_df_slice = review_df_slice[~review_df_slice.index.duplicated(
                keep='first')]

            players = []
            unique_players = list(set(review_df_slice.index))

            for up in unique_players:
                rating, tournament_number, nr_games_played, nr_wins, nr_losses = player_info_lookup(
                    up, score_lookup_table)
                p = uscf_elo.Player(up, rating, nr_games_played,
                                    nr_wins, nr_losses, tournament_number)
                players.append(p)

            combos = list(combinations(review_df_slice.index, 2))
            tournament_results = []
            for c in combos:
                match_result = match_format(c, review_df_slice)
                tournament_results.append(match_result)

            tournament = uscf_elo.Tournament(
                players=players, tournament_results=tournament_results, tournament_date=r)
            if tournament.valid_tournament:
                try:
                    updated_scores = tournament.run()
                    updated_scores_with_reviewer = [
                        v + [u] for v in updated_scores]

                    score_lookup_entry_table = pd.DataFrame(updated_scores_with_reviewer, columns=[
                                                            'wine_id', 'tournament_date', 'tournament_number', 'nr_games_played', 'nr_wins', 'nr_draws', 'nr_losses', 'elo_rating', 'reviewer'])
                    score_lookup_table = score_lookup_table.append(
                        score_lookup_entry_table)
                except:
                    continue

    return score_lookup_table
    # [print(u) for u in updated_scores]
