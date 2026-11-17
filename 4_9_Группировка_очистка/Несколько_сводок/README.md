Фургон с едой сохранил данные о заказах за неделю в файле truck.csv: day, dish, payment и total (доллары). В стартовом коде считывается столбец, по которому нужно сгруппировать данные, в переменную column.

    Выведите количество заказов в каждой группе.
    Выведите одну таблицу со значениями total для каждой группы, со столбцами min, mean и max именно в таком порядке.

Собрать (один раз, скачает крейты csv и serde):
bash

cargo build --release

Успешный конец: Finished \release` profile [optimized] target(s)`.

Запустить, передав колонку группировки через stdin:
bash

printf "day\n" | ./target/release/food_truck


Для входных данных day:

day
Fri    5
Mon    3
Thu    5
Tue    7
Wed    4
dtype: int64
     min   mean  max
day                 
Fri   10  19.60   36
Mon   12  27.00   42
Thu   10  21.60   42
Tue    9  19.00   30
Wed   10  21.25   27
