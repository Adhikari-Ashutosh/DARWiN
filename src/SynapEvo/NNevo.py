import numpy as np
from NN import FFN
class Population:
    def __init__(self , population_size, mutation_rate , input_size , output_size , layers_sizes , nlayers , random_state=None):
        self.size = population_size
        self.mut_rate = mutation_rate
        self.population = [FFN(input_size,output_size,layers_sizes,nlayers) for _ in range(population_size)]
        
    def GetPopN(self):
        return self.population
    
    def Rank(self,Pop):
        scores = np.array([ffn.score for ffn in Pop])
        sorted_indices = np.argsort(scores)[::-1]  # reverse order to get descending
        sorted_ffns = np.array(Pop)[sorted_indices]  # use the indices to sort the array
        return sorted_ffns
    def sort_solutions(self, scores):
        # sort population based on scores
        self.population = [x for _,x in sorted(zip(scores,self.population), reverse=True)]
        
    def select_parents(self):
        # select the best 20% of NN layers as parents
        num_parents = int(self.size * 0.2)
        parents = self.population[:num_parents]
        return parents
    
    def breed_population(self, parents, mutation_rate):
        # create new population by mixing parents' weights and biases with some mutation
        new_population = []
        for i in range(self.size):
            # randomly choose two parents from the selected parents
            parent1, parent2 = np.random.choice(parents, size=2, replace=False)
            
            # create new child by mixing weights and biases of parents
            child_weights = []
            child_biases = []
            for p1_weight, p2_weight in zip(parent1.weights, parent2.weights):
                new_weight = np.random.choice([p1_weight, p2_weight])
                child_weights.append(new_weight)
            for p1_bias, p2_bias in zip(parent1.biases, parent2.biases):
                new_bias = np.random.choice([p1_bias, p2_bias])
                child_biases.append(new_bias)
            
            # add some random mutation
            for weight in child_weights:
                mutation_mask = np.random.choice([0, 1], size=weight.shape, p=[1-mutation_rate, mutation_rate])
                mutation = np.random.normal(scale=0.1, size=weight.shape)
                weight += mutation_mask * mutation
            for bias in child_biases:
                mutation_mask = np.random.choice([0, 1], size=bias.shape, p=[1-mutation_rate, mutation_rate])
                mutation = np.random.normal(scale=0.1, size=bias.shape)
                bias += mutation_mask * mutation
            
            # create new FFN with child weights and biases and add to new population
            child = FFN(parent1.layer_sizes)
            child.weights = child_weights
            child.biases = child_biases
            new_population.append(child)
        self.population = new_population
    
    def evolve(self, RatedPopN):
        self.population = self.rank(RatedPopN)
        parents = self.select_parents()
        self.breed_population(parents, self.mut_rate)