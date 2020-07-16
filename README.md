# SignLanguageRecognition
In diesem Projekt des Maschinellen Lernens wurde mit zwei Hauptdatensätzen die Klassifikation von
Gebärdensprache erprobt. Hierzu wurden die Datensätze ASL-Alphabet (https://www.kaggle.com/grassknoted/asl-alphabet)
und Sign Language MNIST (https://www.kaggle.com/datamunge/sign-language-mnist) genutzt. Es wurden verschiedene
Neuronale Netze und Konvolutionelle Neuronale Netze, unter anderem das VGG19, erprobt, welche alle im logs Ordner
der master Branch zu finden sind. Die Modelle mit den besten Ergebnissen wurden in einer Ausarbeitung dokumentiert
und sind im Ordner ergebnisse_aus_ausarbeitung zu finden. Diese Logs dienen lediglich der Dokumentation und die Python
Dateien in diesen sind nicht ausführbar, da die Umgebung des sign_language_image Pakets fehlt. Wurde eine passende
Python Umgebung erstellt, lassen sich allerdings alle geloggten Python Dateien im sign_language_image Paket ausführen.
Die dokumentierten Modellen sind in diesem Ordner bereits zu finden.
## Installation
Die Anwendung wurde unter Debian 10 und Linux Mint 19 getestet sowie entwickelt. 
Um möglichen Inkompatibilitäten aus dem Weg zu gehen, werden diese als Betriebssystem
empohlen. Weiterhin wird eine Nutzung von Cuda und Tensorflow GPU empfohlen, da einige Modelle auf der CPU
sehr lange optimieren müssen. 
1. Installation von Conda oder einem anderen Python Umgebungsmanager.
2. Installation der nötigen Cuda Umgebung (empohlen).
3. Erstellen einer neuen Python 3.8 Umgebung.
4. Installation der Anforderungen aus der requirements.txt mit PIP.
5. Ausführung beliebiger Python Dateien zur Erstellung von Modellen im sign_language_image Paket möglich.