import numpy as np
import theano
import theano.tensor as tt


#Test de theano et tensor flow
# --------------------------------------------------------------- Prise en main --------------------------------------------------------------- 
# declarer une variable / expression symbolique (TensorVariable):
x = tt.fscalar()
y = tt.fscalar()
z = x + y**2



# compiler une expression symbolique en une fonction avec entrees / sorties (Function):
f = theano.function(inputs=[x, y], outputs=z)



# executer une fonction compilee:
# on affecte la valeur 2 a x et la valeur 3 a y.
# la fonction f renvoie la valeur du calcul contenu dans z
print(f(2, 3))
print '----------------'


# Notez que vous pouvez declarer des variables symboliques munies d'un etat interne persistant (TensorSharedVariable):
x = tt.fscalar()
a = theano.shared(value=1.0)

y = tt.power(x, a)
f = theano.function(inputs=[x], outputs=y)

# ici a = 1, donc 2 puissance 1 = 2
print(f(2))

# on modifie ensuite a en lui donnanrt la valeur 2, ce qui met a jour la fonction
a.set_value(2.0)
print(f(2))
print '----------------'




# L'etat interne d'une variable peut egalement etre mis a jour lors de l'execution d'une fonction compilee, en utilisant l'argument updates:
x = tt.fscalar()
a = theano.shared(value=0.0)
b = theano.shared(value=0.0)

y = tt.power(x, a) + tt.power(x, b)
f = theano.function(inputs=[x], outputs=y, updates=[[a, a + 1.0], [b, b + 2.0]])

# ici, on peut programmer une fonction pour que elle mette a jour a chaque appel les parametres
print(f(2))
print(f(2))
print(f(2))
print '----------------'



# Dans un reseau de neurones classique beaucoup d'operations peuvent etre vues comme des manipulations de scalaires, vecteurs (1D), matrices (2D) et tenseurs (nD).
# Par exemple, le code suivant constitue un modele de type perceptron:
x = tt.fvector()
w = theano.shared(value=np.asarray((1.2, 0.5, -0.2, 0.05, -1.1)))
b = theano.shared(value=0.1)
y = 1 / (1 + tt.exp(-(tt.dot(x, w) + b)))

f = theano.function(inputs=[x],outputs = y)
print (f([1,1,1,1,1]))
print '----------------'

# Repondez aux questions suivantes:	

#    Combien y a-t-il de variables en entree dans ce modele? 
#				5 c'est la taille de x
#    Combien de parametres compte-t-on au total?
#				6, 5 pour la taille de w et le biais b
#    Qu'obtient-on en sortie de ce modele lorsque toutes les entrees valent 1?
#				0.634135591011 avec un tableau de taille 5 de [1 1 1 1 1 ]



# Dans l'exemple precedant, exprimez y en utilisant 'tt.net.sigmoid', et verifiez que vous obtenez bien la meme chose.
y = tt.nnet.sigmoid(tt.dot(x, w) + b)
f = theano.function(inputs=[x],outputs = y)
print (f([1,1,1,1,1]))
print '----------------'

# on obtient bien la meme chose que precedemment

# --------------------------------------------------------------------------------------------------------------------------------------------- 




# ----------------------------------------------------------- Regression logistique ----------------------------------------------------------- 

#Executez et etudiez ce code afin d'en comprendre les differentes etapes:

#    definition du modele;
#    definition de deux fonctions cout (negative log-likelihood, hamming loss);
#    calcul du gradient et de la mise a jour des parametres;
#    compilation des fonctions d'apprentissage et d'evaluation;
#    apprentissage et validation par mini-batches;
#    visualisation de 50 examples mal classes.

# Quel taux d'erreur arrivez-vous a atteindre avec ce modele? --> Hamming Loss 0.075600


