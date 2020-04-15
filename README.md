# epydemy-ai


Using the predictions of the agent-based simulation code [epydemy](https://github.com/nima-siboni/epydemy), we train a deep neural-network to help identifying the individual susceptibile to catching the virus (the high-risk group). This predictions for each individual is based on the following features:
* the number of home-mates for that individual,
* the number of co-workers for that individual, and
* the total number of social-place for that individual.

Using these features, we can find the group which are most at danger, e.g. the individual who have 75% chance of being infected (or more).

The deep neural-network is trained by using almost 4 millions individual in 400 simulated cities, for whom the susceptibility to the virus is turned into a probability. With this data, the trained neural-network is capable of identifying the high-risk group by
* precision = 0.90%
* recall = 0.85%
* F1 = 0.88%

The neural-network is build and trained by TensorFlow 2.0.1 .

List of important files:
* ```preprocessing_data.py``` : converts the raw data of each individual to a probability.
* ```trainer.py``` : builds the neural-network and trains it.

The next step is to use the predictions of this neural-network and **selectively** quarantine some agents in [epydemy](https://github.com/nima-siboni/epydemy) to prevent the pandemic from an outbreak.
