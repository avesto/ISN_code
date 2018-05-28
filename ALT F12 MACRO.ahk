;Raccourci permettant de lancer l'IA avec une seule touche
;-------------------------------------------------
;Script AHK (autohotkey), utilitaire très sympa permettant de faire des macros très puissants
;Ecrit sur Visual Studio Code
;-------------------------------------------------

;La problématique:  en gros, pour lancer ALICE, c'est long: il faut lancer udacity, puis lancer un invite anaconda, se placer dans le bon dossier, se placer dans le bon conda, puis executer le drive.py, ainsi que le modèle généré. Bon, en vrai, c'est pas si long que ca, mais c'est du travail fastidieux qui peux etre automatisé, parce que le minimum de travail humain c'est cool, cf parcoursup
;Pour résumer, AHK va maitriser le clavier et la souris a notre place: quand on le voit, ca fait très IA qui controle le PC, pour rester dans la thématique! (bon, c'est un simple script qu'est pas foutu de faire autre chose hein, c'est pas une IA...)

;-------------------------------------------------

;sorte de directive de préprocesseur, on signale l'utilisation de fonctions propre au positionement d'un curseur selon des coordonnées à l'écran

CoordMode, Mouse, Screen

;-------------------------------------------------

!F12:: ;on mappe notre macro sur alt f12 (le ! renvoie a alt en ahk)

;-------------------------------------------------

;On execute tout d'abord Udacity, et on laisse avec des Sleep des temps d'attente histoire de limiter les erreurs, si 200 trucs se lancent en meme temps ca va pas le faire

Run, "C:\Users\avest\Desktop\ISN\projet isn\Default Windows desktop 64-bit.exe"
Sleep, 2500

;-------------------------------------------------

;On appuie sur entrée, pour faire "play!", et on attend que Udacity ce lance.

Send {Enter}
Sleep, 3000

;-------------------------------------------------

;La... C'est la ou on a un problème "majeur". on doit appuyer sur "autonomous" pour continuer, mais udacity se lance comme un jeu, du coup on ne peut pas naviger avec tab et les flèches. Du coup, on met notre curseur de la souris sur le bouton "autonomous"(position déterminée en faisant BEAUCOUP d'essais...), et on clique gauche. problème majeur? C'était plutot simple en fait!

MouseMove, 750, 530
MouseClick, left
Sleep, 3000

;-------------------------------------------------

;On lance l'invite anaconda tout d'abord. par défaut, il va ouvrir la racine C:\users\avest\desktop, mon bureau quoi. On attend que tout s'initialise bien...

Run, "%windir%\System32\cmd.exe "/K" C:\Users\avest\Miniconda3\Scripts\activate.bat C:\Users\avest\Miniconda3"
Sleep, 5000

;-------------------------------------------------

;On se met dans le bon dossier, qui se situe dans le dossier ISN sur mon bureau, puis dans 'projet isn', et on attend...

Send {Text} cd isn\projet isn
Send, {Enter}
Sleep, 1000

;-------------------------------------------------

;...Puis dans le bon conda...

Send {Text}conda activate car-behavioral-cloning 
Send, {Enter}
Sleep, 1000

;-------------------------------------------------

;...Et on lance la bete!

Send {Text}python drive.py model.h5 
Send, {Enter}
SetKeyDelay 30,50
Send, {ALT DOWN}{TAB}{ALT UP}
Return


