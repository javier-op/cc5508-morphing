# cc5508-morphing

El programa morphing.py es una implementación del algoritmo de morphing de Beier-Neely, para ejectarlo
se necesita Python 3 instalado con las librerías descritas en requirements.txt.

Se ejecuta con la siguiente linea:

python morphing.py --imageA [path to first image] --imageB [path to second image] --lines [path to text file with lines] -n [number of images to generate]

- imageA: imagen de origen
- imageB: imagen de destino
- lines: lineas definidas para el algoritmo
- n: numero de imágenes intermedias a generar

Se incluyen imagenes y lineas definidas de ejemplo en las carpetas sample_imgs y sample_lines respectivamente.
El algoritmo toma bastante tiempo, puede tomar más de una hora, para verificar el funcionamiento se recomienda usar
este ejemplo:

python morphing.py --imageA ./sample_imgs/house1v2.jpg --imageB ./sample_imgs/house2v2.jpg --lines ./sample_lines/lines_houses_v2.txt -n 7

Las imágenes house1v2.jpg y house2v2.jpg son bastante pequeñas por lo que toma alrededor de un minuto generar cada imagen intermedia.
El orden en que se deben poner las imágenes para el resto de los ejemplos y sus lineas son:

- --imageA ./sample_imgs/house1.jpg --imageB ./sample_imgs/house2.jpg --lines ./sample_lines/lines_houses.txt
- --imageA ./sample_imgs/ramsey.jpg --imageB ./sample_imgs/harriott.jpg --lines ./sample_lines/lines_ramsey_harriott.txt
- --imageA ./sample_imgs/freeman.png --imageB ./sample_imgs/devito.png --lines ./sample_lines/lines_freeman_devito.txt

El resultado son dos imágenes con las lineas definidas dibujadas y un gif con las imágenes generadas.
Estas quedan en el mismo directorio en el que se ubica el programa.
