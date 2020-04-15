# Analysis results
# * Turnover = 3422065.0586887286
# * EV = 132712.7898287468
# * PnL = 147777.01029799975
# * Commission = 72974.63452212434
# * NetPnL = 74802.3757758754
# * RoI = 2.186%

# I used python 3.6.8

import pandas as pd
from random import random
import matplotlib.pyplot as plt


def full_kelly(
    probability: float,
    odds: float,
    side: int,
    minimum_stake: float = 2,
    bankroll: float = 10000,
) -> float:
    """
    probability -- estimated probability for the horse to win
    odds -- available odds to back or lay
    side -- 1 to back, 0 to lay
    minimum_stake -- minimum dollar stake admissable
    bankroll -- dollar bankroll available for the bet

    Returns single bet stake as float
    """
    if side == 1:
        stake = (probability * (odds - 1) - (1 - probability)) / (odds - 1) * bankroll
    elif side == 0:
        stake = ((1 - probability) - probability * (odds - 1)) / (odds - 1) * bankroll
    if stake < minimum_stake:
        stake = 0
    return stake


def pnl(stake: float, odds: float, side: int, result: int) -> float:
    """
    stake -- dollar stake
    odds -- available odds to back or lay
    side -- 1 to back, 0 to lay
    result -- 1 if the horse won, 0 otherwise

    Returns single bet pnl as float
    """
    pnl = 0
    if side == 1:
        if result == 1:
            pnl = stake * (odds - 1)
        elif result == 0:
            pnl = stake * -1
    elif side == 0:
        if result == 1:
            pnl = stake * (odds - 1) * -1
        elif result == 0:
            pnl = stake
    return pnl


def EV(stake: float, odds: float, side: int, probability: float) -> float:
    """
    stake -- dollar stake
    odds -- available odds to back or lay
    side -- 1 to back, 0 to lay
    probability -- fair probability for win event

    Returns single bet EV as float
    """
    EV = 0
    if side == 1:
        EV = stake * (odds - 1) * probability - stake * (1 - probability)
    elif side == 0:
        EV = stake * (1 - probability) - stake * (odds - 1) * probability
    return EV


def commissions(
    pnl_results: pd.DataFrame,
    percentage_commissions: float = 0.05,
    column_aggregator: str = "race_number",
    column_pnl: str = "pnl",
) -> float:
    """
    pnl_results -- dataframe with results where "pnl" and "race_number" columns are needed
    percentage_commissions -- percentage commission rate
    column_aggregator -- column used in groupby
    column_pnl -- column with pnl data

    Returns total commissions for pnl_results as float
    """
    if (
        column_aggregator not in pnl_results.columns
        or column_pnl not in pnl_results.columns
    ):
        print(
            "either column_aggregator or column_pnl are missing column from pnl_results dataframe"
        )
        return
    race_pnl = pnl_results.groupby(column_aggregator)[column_pnl].sum()
    commissions = (race_pnl.loc[race_pnl > 0] * percentage_commissions).sum()
    return commissions


def monte_carlo(
    pnl_results: pd.DataFrame, fair_probabilities: str = "fair_probability", trials=10
):
    """
    pnl_results -- the horse dataframe updated with all additional needed columns
    fair_probabilities -- name of the column with fair probabilities to be used in montecarlo simulations
    trails -- number of montecarlo simulations

    Returns trials pnl dataframe and trails commissions dictionary both indexed from 0 to trials -1
    """
    trials_pnl = pd.DataFrame()
    trials_commissions = {}
    for trial_index in range(trials):
        pnl_results["montecarlo_winners"] = (
            pd.Series([random() for x in range(len(pnl_results))])
            < pnl_results[fair_probabilities]
        ) * 1
        pnl_results["montecarlo_pnl"] = pnl_results.apply(
            lambda x: pnl(
                full_kelly(
                    x[fair_probabilities], x["win_starting_price"], x["trade_side"]
                ),
                x["win_starting_price"],
                x["trade_side"],
                x["montecarlo_winners"],
            ),
            axis=1,
        )
        trials_commissions[trial_index] = commissions(
            pnl_results, column_pnl="montecarlo_pnl"
        )
        trials_pnl[trial_index] = pd.Series(pnl_results["montecarlo_pnl"])

    return trials_pnl, trials_commissions


if __name__ == "__main__":
    # import data using pandas
    horses = pd.read_csv("horses.csv")

    # set columns
    horses["fair_probability"] = 1 / horses["win_fair_price"]
    horses["trade_side"] = (horses["win_starting_price"] > horses["win_fair_price"]) * 1
    horses["stake"] = horses.apply(
        lambda x: full_kelly(
            x["fair_probability"], x["win_starting_price"], x["trade_side"]
        ),
        axis=1,
    )

    horses["pnl"] = horses.apply(
        lambda x: pnl(
            full_kelly(x["fair_probability"], x["win_starting_price"], x["trade_side"]),
            x["win_starting_price"],
            x["trade_side"],
            x["winner"],
        ),
        axis=1,
    )
    horses["EV"] = horses.apply(
        lambda x: EV(
            x["stake"], x["win_starting_price"], x["trade_side"], x["fair_probability"]
        ),
        axis=1,
    )

    ## RESULTS
    Turnover = horses["stake"].sum()
    EV = horses["EV"].sum()
    PnL = horses["pnl"].sum()
    Commission = commissions(horses)
    NetPnL = PnL - Commission
    RoI = NetPnL / Turnover
    print("Analysis results")
    print("Turnover = %s" % Turnover)
    print("EV = %s" % EV)
    print("PnL = %s" % PnL)
    print("Commission = %s" % Commission)
    print("NetPnL = %s" % NetPnL)
    print("RoI = %s" % RoI)

    ## MONTECARLO SIMULATION
    # assumptions:
    # the main assumption here is that reported fair probabilities are also the real probabilities
    # This seems to be a decent assumption given the results obtained on the main data comparing EV vs Estimated pnl
    # Montecarlo simulations are obtained looping through trials and comparing fair probabilities
    # vs random [0, 1) numbers that generates different winners every trial.
    #
    # Looking at the chart "montecarlo_net_pnl_distribution_1000_trials" included in the folder,
    # and considering the average and the standard deviation for the net pnl of all the simulations
    # that are respectively 61740.77296605775 and 34912.883933261495,
    # I'd say the strategy is potentially viable and with an average of 1.75% RoI it could still be traded.
    # Left tail shows 3.9% of 1000 simulations ended up being negative with -34713.03916111881 being the minimum.

    # The following snippet of code is actually quite slow (a couple of hours for 1000 trials)
    # I commented it out and just reported the results.
    # If you want to try it, I recommend passing just a few trials.

    # mcr, mcc = monte_carlo(horses, trials=1000)
    # mcr_sum = mcr.sum()
    # montecarlo_net_pnls = pd.Series(
    #     [pnl - mcc[index] for index, pnl in enumerate(mcr_sum)]
    # )
    # montecarlo_net_pnls.hist(bins=50)
    # plt.show()
