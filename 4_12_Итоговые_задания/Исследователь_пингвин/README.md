Программа считывает название столбца.

    Считай penguins.csv; выведи количество пингвинов и количество пропущенных значений в столбце.
    Удали все строки с пропусками и выведи количество оставшихся строк.
    На одной диаграмме пройдись циклом по видам: выведи среднее значение столбца для каждого вида (с 1 десятичным знаком, используя round) и построй его гистограмму с параметрами bins=8, alpha=0.5 и названием вида в качестве метки.
    Задай для диаграммы заголовок со значением столбца, обозначь ось y как Penguins и добавь легенду.

Для bill_length_mm полный вывод выглядит так:

Penguins: 100
Missing bill_length_mm: 2
Kept after cleaning: 94
Adelie: 38.9
Chinstrap: 50.0
Gentoo: 46.9
Charts drawn: 1
Title: bill_length_mm
Y label: Penguins
Histograms: 3
Histogram Adelie: 8 bars, counts [4, 1, 7, 4, 7, 8, 7, 4]
Histogram Chinstrap: 8 bars, counts [1, 2, 5, 2, 5, 4, 0, 1]
Histogram Gentoo: 8 bars, counts [1, 4, 9, 7, 3, 5, 1, 2]
See-through bars: yes
Legend: Adelie, Chinstrap, Gentoo

echo "bill_length_mm" | cargo run --release --quiet
