# epydemy-ai


Using the predictions of possibility of infections by the agent-based simulation code [epydemy](https://github.com/nima-siboni/epydemy) we train a neural-network to help identifying the individual susciptible to catching the virus. This predictions for each individual is based on the following features:
* the number of home-mates for that individual,
* the number of co-workers for that individual, and
* the total number of social-place for that individual.

Using these features, we can find the group which are most at danger, e.g. the individual who have 75% chance of being infected (or more).

The neural-network is trained by using almost 4 millions individual in 400 simulated cities, for whom the susciptilibuy to the virus is turned into a probablity. With this data, the trained neural-network is capable of identifying the high-risk group by
* precision = 0.90%
* recall = 0.85%
* F1 = 0.88%

The neural-network is build and trained by TensorFlow 2.0.1 .
