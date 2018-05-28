import pandas as pd # analyse de données, créer des datasets et jouer avec

import numpy as np #maths maths maths (notamment maths de matrices)


#Voici Keras, notre libraire de deep learning!!
#On inclut avec quelques classes utilitaires (séparer les données d'entraienement et de test, optimisation, sauvegarde, type de couches, génération d'images d'entrainement avec données respectivement)
from sklearn.model_selection import train_test_split 
from keras.models import Sequential
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from keras.layers import Lambda, Conv2D, MaxPooling2D, Dropout, Dense, Flatten
from utils import INPUT_SHAPE, batch_generator

#arguments de ligne de commande
import argparse

#Lire des fichiers
import os

#pour du débgugage, pour permettre d'avoir des résultats reproductibles
np.random.seed(0)


def load_data(args):


    """
  BUT DE CETTE FONCITON:  Charger les données d'entrainement et les séparer en données d'entrainement et de validation
    """



    #lire les fichiers CSV  en une variable de dataframe
    data_df = pd.read_csv(os.path.join(os.getcwd(), args.data_dir, 'driving_log.csv'), names=['center', 'left', 'right', 'steering', 'throttle', 'reverse', 'speed'])


    #on sauvegarde les images des cameras en tant que nos données d'entrée
    X = data_df[['center', 'left', 'right']].values


    #...et nos données du volant (position angulaire) comme notre data d'entrée
    y = data_df['steering'].values


    #Maintenant on peut séparer les données en des sets d'entrainement (80%), de test (20%) et de validation, on dit merci scikit learn!!!
    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=args.test_size, random_state=0)


    return X_train, X_valid, y_train, y_valid


def build_model(args):
    """
    BUT: le modèle d'analyse d'images
 MODELE DU NVIDIA PX1 (infos piquées sur leur présentation sur le sujet)

 "
    Image normalization to avoid saturation and make gradients work better.
    Convolution: 5x5, filter: 24, strides: 2x2, activation: ELU
    Convolution: 5x5, filter: 36, strides: 2x2, activation: ELU
    Convolution: 5x5, filter: 48, strides: 2x2, activation: ELU
    Convolution: 3x3, filter: 64, strides: 1x1, activation: ELU
    Convolution: 3x3, filter: 64, strides: 1x1, activation: ELU
    Drop out (0.5)
    Fully connected: neurons: 100, activation: ELU
    Fully connected: neurons: 50, activation: ELU
    Fully connected: neurons: 10, activation: ELU
    Fully connected: neurons: 1 (output)

    the convolution layers are meant to handle feature engineering
    the fully connected layer for predicting the steering angle.
    dropout avoids overfitting
    ELU(Exponential linear unit) function takes care of the Vanishing gradient problem. 
 "
  
 
    """


    #Modèle d'analyse d'image, réseau neuronal en utilisant les données de Nvidia
    #En gros, on créé plusieurs couches de covolotion. A chaque fois, on réduit la taille des
    #couches en les faisant passer dans des filtres. A la fin la couche est tellement fine
    #qu'on finit avec... Une seule valeur finale, notre modèle! 

    model = Sequential()
    model.add(Lambda(lambda x: x/127.5-1.0, input_shape=INPUT_SHAPE))
    model.add(Conv2D(24, 5, 5, activation='elu', subsample=(2, 2)))
    model.add(Conv2D(36, 5, 5, activation='elu', subsample=(2, 2)))
    model.add(Conv2D(48, 5, 5, activation='elu', subsample=(2, 2)))
    model.add(Conv2D(64, 3, 3, activation='elu'))
    model.add(Conv2D(64, 3, 3, activation='elu'))
    model.add(Dropout(args.keep_prob))
    model.add(Flatten())
    model.add(Dense(100, activation='elu'))
    model.add(Dense(50, activation='elu'))
    model.add(Dense(10, activation='elu'))
    model.add(Dense(1))
    model.summary()

    return model


def train_model(model, args, X_train, X_valid, y_train, y_valid):
    """
    Ici, la fonction va entrainer le modèle généré par le code ci dessus
    """
    

    """
    Ici, on fait quoi?
    1*/ On sauvegarde le modèle après chaque période
    2*/ On prend une sorte de quantité à surveiller, la verbosité (un sorte de log qui prend pour valeur 0 ou 1)
    3*/ A chaque génération, on vérifie si la verbosité est meilleure que celle de la génération précédente, si c'est le cas, on écrase la verbosité précédente avec l'actuelle.

    on créé un checkpoint, un sorte de point de sauvegarde par époque, et on lui donne plusieurs paramètres:
       >mode: soit auto, min ou max. on a choisi auto pour une simple question d'aisance. 
       >verbose: notre quantité a vérifier
       >save_best_only: ce qui va nous permettre de vérifier que de la verbosité d'une génération est meilleure que la précédente
    """
    checkpoint = ModelCheckpoint('model-{epoch:03d}.h5',
                                 monitor='val_loss',
                                 verbose=0,
                                 save_best_only=args.save_best_only,
                                 mode='auto')






    """
    Ici:
    On calcule la différence entre l'angle de rotation attendu et l'angle de rotation réel
    On le met au carré
    On addditionne toutes les différences pour avoir un maximum de points de données
    On va diviser cette somme par l'effectif (le nb de différences)
    et cette valeur magique, c'est une sorte de moyenne, c'est ce que l'on va réduire avec Adam
    """
    model.compile(loss='mean_squared_error', optimizer=Adam(lr=args.learning_rate))
    






    """
    Notre modèle du coup, on l'adapte pour des données générées par un générateur en lots par lots (de données)
    (Fun fact: pour plus d'efficacité, le générateur est executé en parallèle du modèle)
    (il est possible de faire tourner le modèle et le générateur sur le GPU afin de laisser le CPU libre de faire d'autres taches)
    """
    model.fit_generator(batch_generator(args.data_dir, X_train, y_train, args.batch_size, True),
                        args.samples_per_epoch,
                        args.nb_epoch,
                        max_q_size=1,
                        validation_data=batch_generator(args.data_dir, X_valid, y_valid, args.batch_size, False),
                        nb_val_samples=len(X_valid),
                        callbacks=[checkpoint],
                        verbose=1)





#Fonction utilitaire gérant les arguments de ligne de commande
#  (Fournie par Udacity)
def s2b(s):
    """
    Simple conversion d'une chaine de caractères (abcdefg 1234) en booléeenne (1 ou 0)
    """
    s = s.lower()
    return s == 'true' or s == 'yes' or s == 'y' or s == '1'


def main():
    
    """
    Le gros de notre code: charger les données d'entrainement et de validation et entrainner le modèle. 
    """
    
    parser = argparse.ArgumentParser(description='A.L.I.C.E, Artificial Learning Intelligence for Car Emulation')
    parser.add_argument('-d', help='Destination des donnees',        dest='data_dir',          type=str,   default='data')
    parser.add_argument('-t', help='taille du lot de test',    dest='test_size',         type=float, default=0.2)
    parser.add_argument('-k', help='proba de l abandon du calcul,  dest='keep_prob',         type=float, default=0.5)
    parser.add_argument('-n', help='nombre de periodes',      dest='nb_epoch',          type=int,   default=10)
    parser.add_argument('-s', help='echantillons par periodes',     dest='samples_per_epoch', type=int,   default=20000)
    parser.add_argument('-b', help='taille du lot',            dest='batch_size',        type=int,   default=40)
    parser.add_argument('-o', help='enregistrement du meilleur modele', dest='save_best_only',    type=s2b,   default='true')
    parser.add_argument('-l', help='rythme d apprentissage',         dest='learning_rate',     type=float, default=1.0e-4)
    args = parser.parse_args()



    #Afficher nos paramètres
    print('-' * 30)
        print('Paramètres')
        print('-' * 30)
    for key, value in vars(args).items():
        print('{:<20} := {}'.format(key, value))
    print('-' * 30)

    #Charger nos données
    data = load_data(args)


    #construire le modèle
    model = build_model(args)


    #entrainer le modèle (le fichier généré s'appelle modèle.h5)
    train_model(model, args, *data)


if __name__ == '__main__'
    main()



