# tradingPlatform

I've always been interested in stocks and trading but I had never attempted any sort of project in it. So after some reading about different concepts and picking a route I wanted to go down I decided to commit to building a trading platform. What I mean by a trading platform is a system that includes a strategy and a way of backtesting this strategy. And well, I've finally got a working a platform. It's far from perfect, this is only really the beginning but it has a functioning strategy and a backtests it to get details of its performance. 

What is the strategy used?

I discovered the concept of pair trading and cointegration by accident. But it instantly struck me as an interesting introductory project for me as it had a nice amount of statistics in it. I've decided to start off with the most basic form of a cointegration pairs trading strategy I could think of check if a pair is cointegrated over the last n months, if it is then over the next k months, long the spread when it is low, and short the spread when it is high, closing positions as it approaches the mean. 

For more details about a similar strategy see this video:
https://www.quantopian.com/lectures/introduction-to-pairs-trading
