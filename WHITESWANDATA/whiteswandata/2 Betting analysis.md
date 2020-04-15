Betting analysis
================


1 Instructions
--------------

Please complete the accompanying betting exercises before starting this analysis.

You might have to look up terms and techniques that are unfamiliar to you.

You must use Python and libraries that are designed for data analysis and scientific programming. The Python-library *pandas* and *numpy* are preferred.

You must use vectorised operations everywhere that they are practical.

We want to see the source code that you write to do this analysis. Your source code must be a plain-text file. A Jupyter notebook is preferred.

Please write plenty of comments in your source code:
* What you want to do.
* Decisions and assumptions that you make.
* What you do not know or understand.

Programming guidelines: (*this is very important!*)
* Your ability to write clean, reusable code will be assessed as well as the correctness of your analysis.
* Your code must meet professional standards. We want to see how your style and approach to programming fit with our team. We like type hints, docstrings, pep8 etc 
* Please create a requirements.txt file with your libraries & versions so that one could easily run pip install -r requirements.txt inside a virtual environment.

2 Data
------

Use the horse-racing data in the accompanying file `horses.csv`.

The fields in `horses.csv` are:
* Race number.
* Saddle number.
* Win fair price: Our estimate of the fair price (before the start) for the horse to win the race.
* Win starting price: The starting price for the horse to win the race.
* Winner: Whether the horse won the race (1) or not (0).


3 Analysis
----------

The project:
* You bet at the starting prices. Assume that you know the starting prices before you bet (because, in reality, you do not know them).
* Assume you receive a data feed like the horses.csv on a daily basis. Make sure that your code is robust and could be run automatically without supervision.

Calculate your stake and side (back or lay) for each horse at the starting price using the Kelly criterion and our estimate of the fair price:
* Treat each horse in each race independently.
* Bet full Kelly.
* Use a bank of £10,000 for each horse. Do not update the bank with your PnL.
* Do not bet if your calculated stake is less than the minimum stake of £2.

Calculate the following and write their total values in a comment at the top of your source code:
* Turnover.
* EV.
* PnL.
* Commission, which is 5% of your race PnL in each race that your race PnL is positive.
* Net PnL, which is PnL - commission.
* RoI, which is net PnL / turnover * 100%.

How confident are you that this project has an edge? Do a Monte Carlo experiment.
