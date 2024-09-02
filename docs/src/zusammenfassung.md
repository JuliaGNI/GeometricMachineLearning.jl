# Zusammenfassung

Wissenschaftliches Rechnen ist ein unverzichtbares Werkzeug für viele Disziplinen wie Biologie, Ingenieurwesen und Physik. Es ist wichtig für z.B. (i) die Verknüpfung zwischen Theorien und empirischen Beobachtungen, (ii) Vorhersagen ohne physische Experimente zu treffen und (iii) das Design von Apparaturen wie Fusionsreaktoren. In der Praxis bedeutet wissenschaftliches Rechnen fast immer dass partielle Differentialgleichungen gelöst werden müssen, meist auf Supercomputern; und das ist gewöhnlich sehr teuer. 

Wissenschaftler versuchen seit langem die Kosten für das Lösen dieser Gleichungen zu senken indem sie z. B. Modelle vereinfachen oder reduzierte Darstellungen konstruieren. Einer dieser Ansätze ist *Data-Driven Reduced Order Modeling* (DDROM), das in den letzten Jahren aufgrund seiner Eignung für moderne Hardware und Algorithmen an Bedeutung gewonnen hat. In der Praxis bedeutet das oft die Anwendung von Techniken des maschinellen Lernens wie neuronale Netze.

Neuronale Netze haben in verschiedenen Anwendungen ein enormes Potenzial gezeigt. Für DDROM sind die entsprechenden Ergebnisse oft allerdings noch nicht sehr zufriedenstellend. Wissenschaftler haben in ihrer Anwendung von neuronalen Netzen oft Eigenschaften vernachlässigt, die sich im traditionellen wissenschaftlichen Rechnen als sehr wichtig erwiesen haben. In dieser Arbeit beziehen sich diese Eigenschaften auf die Struktur der Differentialgleichungen; diese ist bei der Durchführung von Simulationen oft unverzichtbar um Stabilität zu gewährleisten.

In dieser Arbeit bezeichnen wir Methoden des maschinellen Lernens, die auf die spezifische Struktur einer Differentialgleichung zugeschnitten sind, als *geometrisches maschinelles Lernen*. Der Begriff *geometrisch* wird in diesem Zusammenhang traditionell auch verwendet und ist als Synonym des Wortes *strukturerhaltend* zu verstehen. Die Idee neuronale Netze *geometrisch* zu machen ist nicht neu; viele der Netzwerke die wir hier presentieren sind es aber und bilden einen wichtigen Teil dieser Dissertation. Diese Arbeit teilt sich in vier Teile.

In Teil I geben wir Hintergrundinformationen wieder, die keine neue Arbeit darstellen, aber die Grundlage für die folgenden Kapitel bilden. Dieser erste Teil enthält eine grundlegende Einführung in die Theorie der Riemannschen Mannigfaltigkeiten, eine Diskussion über Strukturerhaltung und eine kurze Erläuterung von DDROM.

In Teil II wird ein neues Optimierungsframework eingeführt, das bestehende Optimierer für neuronale Netze auf Mannigfaltigkeiten verallgemeinert. Beispiele hierfür sind der Adam-Optimierer und der BFGS-Optimierer. Diese neuen Optimierer waren notwendig, um das Training einer neuen neuronalen Netzwerkarchitektur zu ermöglichen, die wir *symplektische Autoenkoder* (SAE) nennen.

In Teil III werden schließlich verschiedene spezielle neuronale Netzwerkarchitekturen. Einige von ihnen, wie *SympNets* und *Multi-Head Attention*, stellen keine Neuheiten da, aber andere, wie SAEs, *Volume-Preserving Attention* und der *lineare symplektische Transformer*, sind originell.

In Teil IV geben wir einige Ergebnisse an, die auf den neuen Architekturen basieren. Die meisten dieser Anwendungen beziehen sich auf Anwendungen aus der Physik; um jedoch die neuen Optimierer zu demonstrieren, greifen wir auf ein klassisches Problem aus der Bildklassifikation zurück. Wir wollen damit zeigen dass geometrisches maschinelles Lernen auch in Bereichen außerhalb des wissenschaftlichen Rechnens Anwendung finden kann. In allen behandelten Problemen ist zu sehen dass unsere Modelle genauer oder schneller als vergleichbare Architekturen sind. In einem Beispiel zeigen wir wie ein SAE-reduziertes Model die Auswertung eines Problems um einen Faktor 1000 beschleunigt.

```@raw latex
\clearpage
```