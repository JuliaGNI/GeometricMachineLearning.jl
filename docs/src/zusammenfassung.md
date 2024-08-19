# Zusammenfassung

Wissenschaftliches Rechnen ist ein unverzichtbares Werkzeug für viele Disziplinen wie Biologie, Ingenieurwesen und Physik. Es ist wichtig für z.B. (i) die Verknüpfung zwischen Theorien und Experimenten, (ii) Vorhersagen ohne physische Experimente zu treffen und (iii) das Design von Apparaturen wie Fusionsreaktoren. In der Praxis bedeutet wissenschaftliches Rechnen fast immer dass partielle Differentialgleichungen gelöst werden müssen, meist auf Supercomputern; und das ist sehr teuer. 

Wissenschaftler versuchen seit langem die Kosten für das Lösen dieser Gleichungen zu senken indem sie z. B. Modelle vereinfachen oder reduzierte Darstellungen konstruieren. Einer dieser Ansätze ist *Data-Driven Reduced Order Modeling* (DDROM), das in den letzten Jahren aufgrund seiner Eignung für moderne Hardware und Algorithmen an Bedeutung gewonnen hat. In der Praxis bedeutet das oft die Anwendung von Techniken des maschinellen Lernens sowie neuronale Netzwerke.

Neuronale Netze haben in verschiedenen Anwendungen ein enormes Potenzial gezeigt. Für DDROM sind die entsprechenden Ergebnisse oft allerdings noch nicht sehr zufriedenstellend. Wissenschaftler haben in ihren Algorithmen oft Eigenschaften vernachlässigt, die sich im traditionellen wissenschaftlichen Rechnen als sehr wichtig erwiesen haben. In dieser Arbeit beziehen sich diese Eigenschaften auf die Struktur der Differentialgleichungen; diese ist bei der Durchführung von Simulationen oft unverzichtbar um Stabilität zu gewährleisten.

In dieser Arbeit bezeichnen wir maschinelles Lernen, die auf die spezifische Struktur einer Differentialgleichung zugeschnitten sind, als *geometrisches maschinelles Lernen*. Der Begriff *geometrisch* wird in diesem Zusammenhang traditionell auch verwendet und ist als Synonym des Wortes *strukturerhaltend* zu verstehen. Die Idee neuronale Netze *geometrisch zu machen* ist nicht neu, viele der Netzwerke die wir hier presentieren sind es aber und bilden einen wichtigen Teil dieser Dissertation.

In Teil I geben wir Hintergrundinformationen, die keine neue Arbeit darstellen, aber die Grundlage für die folgenden Kapitel bilden. Dieser erste Teil enthält eine grundlegende Einführung in die Theorie der Riemannschen Mannigfaltigkeiten, eine grundlegende Diskussion der Strukturerhaltung und eine kurze Erläuterung der Modellierung reduzierter Ordnung.

In Teil II wird ein neues Optimierungsframework eingeführt, das bestehende Optimierer für neuronale Netze auf Mannigfaltigkeiten verallgemeinert. Beispiele hierfür sind der Adam-Optimierer und der BFGS-Optimierer. Diese neuen Optimierer waren notwendig, um das Training einer neuen neuronalen Netzwerkarchitektur zu ermöglichen, die wir *symplektische Autokoder* nennen.

In Teil III werden schließlich verschiedene spezielle neuronale Netzwerkarchitekturen. Einige von ihnen, wie *SympNets* und *Multi-Head Attention*, stellen keine Neuheiten da, aber andere, wie *Volume-Preserving Attention* und der *lineare symplektische Transformer*, sind originell.

In Teil IV geben wir einige Ergebnisse an, die auf den neuen Architekturen basieren. Die meisten dieser Anwendungen beziehen sich auf Anwendungen aus der Physik; um jedoch die neuen Optimierer zu demonstrieren, greifen wir auf ein klassisches Problem aus der Bildklassifikation zurück, um zu zeigen, dass geometrisches maschinelles Lernen auch in Bereichen außerhalb des wissenschaftlichen Rechnens Anwendung finden kann.

```@raw latex
\clearpage
```