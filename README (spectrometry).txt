Consideraciones para la interpretación de la espectroscopía

spectrometry_analyzer y voigt_profile utilizan el mismo método para calcular las áreas, pero son independientes. Se usa este último para visualizar y corroborar la correcta asignación de longitudes de onda y cálculo de área.

------------------------------------------------------------------------------------------------------------------------------------------
voigt_profile:

-No se consideran puntos bajo 25 A.U. (valor arbitrario), una vez restado el ruido de fondo, para la identificación de peaks debido a que pueden ser confundidos por el ruido remanente.

-Hombros muy tenues, pero visibles, en ciertos peak son identificados algunas veces y otras no en distintos frames para el mismo shot. Al tener su señal muy débil, interpretarlos como tal; no su existencia y desaparición alternada. Es decir, no se logra determinar la evolución temporal de estos, solo su presencia.

-Hombros con buena intensidad pero opacados por el peak principal (alto A.U. pero parecen más una desviación en la caída del peak más intenso) logran ser identificados en la mayoría de casos, no en todos. En los casos fallidos, estos pueden desviar el centroide del peak principal. Tenerlos en cuenta y registrarlos para futuras correcciones. Ej: 481.25 nm usualmente opaca un peak en 482.16 nm.

-La caracterización del peak principal de Helio 587 nm suele presentar complicaciones debido a saturación del espectrómetro. Código no enfocado en solucionarlo.

-Se aceptan correcciones de optimización :)

------------------------------------------------------------------------------------------------------------------------------------------
spectrometry_analyzer:

-La asignación de iones a los peaks identificados se realiza principalmente con una lista de iones previamente registrados (obtenida de GOLEM). En caso de observar un peak fuera de esta lista, se utiliza el archivo NIST para asignarle un/unos ión/es. En el caso donde no se le logre identificar con las fuentes disponibles, se le etiquetará como "Unknown".

-Se etiqueta a un peak con más de un ión únicamente porque estos son similares en longitud de onda (Ej: "C I/Fe I/N II"). La interpretación final de cuál ion está realmente aportando esa emisión debe realizarse considerando el contexto físico del disparo.

-La longitud de onda mostrada en la etiqueta de un ion corresponde a su valor teórico tabulado, no al centroide exacto medido en el espectro. Debido a la calibración del espectrómetro, el pico real analizado puede presentar un desfase instrumental de hasta 0.7 nm respecto al valor de la etiqueta.

-Se aceptan correcciones de optimización :)

------------------------------------------------------------------------------------------------------------------------------------------
main_app:

-La resolución temporal de la curva está determinada por el tiempo de integración del espectrómetro registrado, el cual puede cambiar por shot. Cada punto está graficado donde terminó la recolección de señal. Ej: Frame 0 (0 - 2 ms) se grafica en 2 ms.

-Se inicia graficando la evolución temporal de las líneas de emisión más pronunciadas. En el panel de iones quedan registrados el resto de iones.

-Recordar que cada punto representa el área bajo la curva de las líneas de emisión, no su valor máximo.

-El análisis de las curvas debe ir de la mano con la visualización de los espectros crudos.

-Se aceptan correcciones de optimización :)






