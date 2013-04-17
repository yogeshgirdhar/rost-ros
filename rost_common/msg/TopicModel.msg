#number of topics
int32 K

#number of vocabulary words
int32 V

#Dirichlet prior topic distribution in a cell
float32 alpha

#Dirichlet prior word distribution in a topic
float32 beta

#flattened KxV matrix which stores the word distribution for each topic
#phi[k*V + v] is the frequency of vocabulary word <v> in topic <k>
int32[] phi

#topic_weights[k] is proportional to probability of seeing topic[k]
int32[] topic_weights

