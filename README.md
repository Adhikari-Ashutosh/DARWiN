# SynapEvo
SynapEvo: Synaptic Evolution
 🙌An attempt to create a Evolutionary neural net library, will have all the numba optimizations for quicker processing!
 A copy of the darwinian theorems for creating better networks per generation.
 ---
 USAGE:

Alrighty folks the code has been completed and it's on pip 🙌!
```
pip install SynapEvo
```
So I must say there's some sneaky ways we've gotten around problems which may/may not be innovative.
Let's have a look at test.py
You can possibly run this code using (to see if you install worked)
```
import SynapEvo.test
```


--- 
Here we have a simple Input data set that represents binary input of size 2:
```
input_set = [[0,0],[0,1],[1,0],[1,1]]
```
This represents our training input data! Basically read as x0,x1

We have our output data as well:
```
output_set = [0,1,1,1]
```
Yes you guessed it! It's an OR logic gate. 🥳

We want SynapEvo to return a Solution that is the best possible output that we could possibly have!

Let's import our Population module! 👍
```
from SynapEvo.NNevo import Population
```
and use this to define a population with some parameters! (These parameters will make sense to you if you know about feed forward NNs)
```
Pop = Population(input_size = 2, output_size = 1 , 
                 layers_sizes = 2 , nlayers = 3, 
                 np_nr = 0.70 ,population_size = 500,
                 parent_percentage = 0.10,mutation_rate = 0.2)
```
But let me just explain them in brief 😁:
- input_size: What is the Size of the array that you will be putting in the model?
- output_size: What is the size you expect out of the Model?
- layers_sizes: How many nodes do you want in one layer of the network?
- nlayers: How many layers do you want in your model?
- np_nr: What is the ratio of the parent offspring(Best performing species crossing over) to random offspring that you want to keep in a new population?
- population_size: How many solutions do you want to be in one generation?
- parent_percentage: What percent(between 0 and 1) of the parents should be kept alive?
- mutation_rate: What is the probability that a child born would have a certain mutation?

Alright! we've initialized our population! Now we need to get a list of of it back to run a for loop over them!

Simply:
```
ourpop = Pop.get_populations()
```
And voila! you have a list of NNs! 🎊

Alright all you have to do now is to assign a score to the objects inside the list and run the evolve function as such!
```
for loop in ourpop:
    output = ourpop[i]
    ourpop[i].score = your_eval_function(output)
ourpop = Pop.evolve(ourpop)
```
Now run this for as many generations as you like and you will get a solution that fits the bucket, at a point! 😅

---
Let's look at the complete code now!
```
from SynapEvo.NNevo import Population
import matplotlib.pyplot as plt
input_set = [[0,0],[0,1],[1,0],[1,1]]
output_set = [0,1,1,1]
Pop = Population(input_size = 2, output_size = 1 , 
                 layers_sizes = 2 , nlayers = 3, 
                 np_nr = 0.70 ,population_size = 500,
                 parent_percentage = 0.10,mutation_rate = 0.2)

ourpop = Pop.get_populations()
aths = 0
athss = None
hsgen=[]
for i in range(500):
    # running for about 500 generations
    hs = 0
    for j in range(len(ourpop)):
        score = 4.0
        for k in range(len(input_set)):
            res = ourpop[j].forward(input_set[k])
            score -=  (res - output_set[k])**2
        ourpop[j].score = score
        if score>hs:
            hs = score
        if score > aths:
            aths = score
            athss = ourpop[j]    
    hsgen.append(hs) 
    ourpop = Pop.evolve(ourpop)
    
for i in input_set:
    print(athss.forward(i))
print(aths)
plt.plot(hsgen)
plt.show()
```
Hope this helps!
