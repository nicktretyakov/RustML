use plotly::common::{Font, Mode};
use plotly::layout::{Annotation, Axis, GridPattern, Layout, LayoutGrid};
use plotly::{Plot, Scatter};

/// Аналог matplotlib-функции label(): заголовок подграфика через аннотацию.
fn label_annotation(text: &str, x: f64) -> Annotation {
    Annotation::new()
        .text(text)
        .x(x)
        .y(1.08)
        .x_ref("paper")   // ← было xref
        .y_ref("paper")   // ← было yref
        .show_arrow(false)
        .font(Font::new().size(16)) // ← было 16.0
}

fn main() {
    let hikes: Vec<f64> = (1..=8).map(|x| x as f64).collect();
    let km = vec![8.0, 12.5, 6.0, 15.0, 10.0, 18.5, 9.5, 14.0];
    let hours = vec![2.5, 4.0, 2.0, 5.0, 3.5, 6.5, 3.0, 4.5];

    let line_trace = Scatter::new(hikes, km.clone())
        .mode(Mode::Lines)
        .name("Distance");

    let scatter_trace = Scatter::new(km, hours)
        .mode(Mode::Markers)
        .name("Time vs Distance")
        .x_axis("x2")
        .y_axis("y2");

    let mut plot = Plot::new();
    plot.add_trace(line_trace);
    plot.add_trace(scatter_trace);

    let grid = LayoutGrid::new()
        .rows(1)
        .columns(2)
        .pattern(GridPattern::Independent);

    let layout = Layout::new()
        .grid(grid)
        .x_axis(Axis::new().title("Hike"))
        .y_axis(Axis::new().title("Distance (km)"))
        .x_axis2(Axis::new().title("Distance (km)"))
        .y_axis2(Axis::new().title("Time (h)"))
        .annotations(vec![
            // ← метод .annotations() принимает Vec<Annotation>
            label_annotation("Distance per Hike", 0.22),
            label_annotation("Time vs Distance", 0.78),
        ]);

    plot.set_layout(layout);
    plot.show();
}
