# tradingPlatform

I've always been interested in stocks and trading but I had never attempted any sort of project in it. So after some reading about different concepts and picking a route I wanted to go down I decided to commit to building a trading platform. What I mean by a trading platform is a system that includes a strategy and a way of backtesting this strategy. And well, I've finally got a working a platform. It's far from perfect, this is only really the beginning but it has a functioning strategy and a backtester.

What is the strategy used?

I discovered the concept of pair trading and cointegration by accident. But it instantly struck me as an interesting introductory project for me as it had a nice amount of statistics in it. I've decided to start off with the most basic form of a cointegration pairs trading strategy I could think of check if a pair is cointegrated over the last n months, if it is then over the next k months, long the spread when it is low, and short the spread when it is high, closing positions as it approaches the mean. 

Code layout:

I’ve broken the code into 4 different sections: <br>
- Getting the data (DataManager.py) <br>
- Analysing the data for cointegration (cointAnalysis.py) <br>
- Creating the strategy (cointStrategy.py) <br>
- And finally, backtesting (backtester.py). <br>


Future Plan:

- Implement a stop loss <br>
- Use rolling windows instead of fixed <br>
- Baskets instead of pairs <br>
- Look for cointegration over several windows not just one <br>
- Hurst exponent (or Half Life) <br>


For more details about a similar strategy see this video:
https://www.quantopian.com/lectures/introduction-to-pairs-trading
