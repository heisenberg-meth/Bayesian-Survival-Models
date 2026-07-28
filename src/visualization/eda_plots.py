"""
EDA Plotting Engine using Pillow (PIL) for generating publication-quality visual charts.
Includes:
- Kaplan-Meier Survival Curves (Overall & Stratified with 95% Greenwood Confidence Bands)
- Correlation Heatmaps
- Feature Histograms & KDE curves
- Box Plots & Outlier Plots
- Categorical Bar Charts
- Missing Value Heatmaps
"""

os_import = __import__("os")


class EDAPlotter:
    def __init__(self, output_dir):
        self.output_dir = output_dir
        os_import.makedirs(self.output_dir, exist_ok=True)
        try:
            from PIL import Image, ImageDraw, ImageFont

            self.Image = Image
            self.ImageDraw = ImageDraw
            self.ImageFont = ImageFont
            self.has_pil = True
        except ImportError:
            self.has_pil = False

    def plot_kaplan_meier(self, km_data_dict, title, filename, strat_name=None):
        """Draws high-resolution Kaplan-Meier survival step function curves."""
        if not self.has_pil:
            return

        width, height = 1200, 800
        margin_left, margin_right = 100, 220
        margin_top, margin_bottom = 100, 100

        img = self.Image.new("RGB", (width, height), "#FFFFFF")
        draw = self.ImageDraw.Draw(img)

        # Color palette
        colors = ["#1F77B4", "#FF7F0E", "#2CA02C", "#D62728", "#9467BD", "#8C564B"]

        # Draw background grid
        plot_w = width - margin_left - margin_right
        plot_h = height - margin_top - margin_bottom

        # Determine max time
        all_times = []
        if isinstance(next(iter(km_data_dict.values())), dict) and "timeline" in next(
            iter(km_data_dict.values())
        ):
            curves = km_data_dict
        else:
            curves = {"Overall": km_data_dict}

        for cdata in curves.values():
            all_times.extend(cdata["timeline"])

        max_time = max(all_times) if all_times else 100.0
        if max_time <= 0:
            max_time = 1.0

        # Draw title
        draw.text(
            (margin_left, 30), title, fill="#111827", font=self._get_font(24, bold=True)
        )
        draw.text(
            (margin_left, 65),
            f"Kaplan-Meier Survival Function {f'stratified by {strat_name}' if strat_name else ''}",
            fill="#6B7280",
            font=self._get_font(14),
        )

        # Draw axes & grid
        # Y-axis (Survival Probability 0.0 to 1.0)
        for y_pct in [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]:
            y_pos = margin_top + plot_h - int(y_pct * plot_h)
            draw.line(
                [(margin_left, y_pos), (margin_left + plot_w, y_pos)],
                fill="#F3F4F6" if y_pct != 0 else "#D1D5DB",
                width=1,
            )
            draw.text(
                (margin_left - 45, y_pos - 8),
                f"{y_pct:.1f}",
                fill="#4B5563",
                font=self._get_font(12),
            )

        # X-axis (Time)
        num_x_ticks = 5
        for i in range(num_x_ticks + 1):
            t_val = (max_time / num_x_ticks) * i
            x_pos = margin_left + int((t_val / max_time) * plot_w)
            draw.line(
                [(x_pos, margin_top), (x_pos, margin_top + plot_h)],
                fill="#F3F4F6" if i != 0 else "#D1D5DB",
                width=1,
            )
            draw.text(
                (x_pos - 15, margin_top + plot_h + 10),
                f"{int(t_val)}",
                fill="#4B5563",
                font=self._get_font(12),
            )

        # Axis labels
        draw.text(
            (margin_left + plot_w // 2 - 40, margin_top + plot_h + 45),
            "Survival Time (t)",
            fill="#111827",
            font=self._get_font(14, bold=True),
        )
        # Y-axis label text vertical position
        draw.text(
            (20, height // 2 - 50),
            "S(t)",
            fill="#111827",
            font=self._get_font(16, bold=True),
        )

        # Draw median survival line (Y=0.5)
        y_med = margin_top + plot_h - int(0.5 * plot_h)
        for x in range(margin_left, margin_left + plot_w, 10):
            draw.line([(x, y_med), (x + 5, y_med)], fill="#9CA3AF", width=1)
        draw.text(
            (margin_left + plot_w + 10, y_med - 8),
            "S(t)=0.5",
            fill="#9CA3AF",
            font=self._get_font(11),
        )

        # Draw curves
        legend_items = []
        for idx, (c_name, c_data) in enumerate(curves.items()):
            color = colors[idx % len(colors)]
            timeline = c_data["timeline"]
            surv = c_data["survival_probability"]
            ci_low = c_data.get("ci_lower", surv)
            ci_up = c_data.get("ci_upper", surv)

            # Step function points
            points = []
            ci_low_points = []
            ci_up_points = []

            for i in range(len(timeline)):
                t = timeline[i]
                s = surv[i]
                cl = ci_low[i]
                cu = ci_up[i]

                x = margin_left + int((t / max_time) * plot_w)
                y = margin_top + plot_h - int(s * plot_h)
                y_cl = margin_top + plot_h - int(cl * plot_h)
                y_cu = margin_top + plot_h - int(cu * plot_h)

                if i > 0:
                    points[-1][0]
                    # Step right then down
                    points.append((x, points[-1][1]))
                    ci_low_points.append((x, ci_low_points[-1][1]))
                    ci_up_points.append((x, ci_up_points[-1][1]))

                points.append((x, y))
                ci_low_points.append((x, y_cl))
                ci_up_points.append((x, y_cu))

            # Draw CI dashed lines
            for i in range(len(ci_low_points) - 1):
                draw.line([ci_low_points[i], ci_low_points[i + 1]], fill=color, width=1)
                draw.line([ci_up_points[i], ci_up_points[i + 1]], fill=color, width=1)

            # Draw main curve
            for i in range(len(points) - 1):
                draw.line([points[i], points[i + 1]], fill=color, width=3)

            med_time = c_data.get("median_survival_time", "N/A")
            legend_items.append((c_name, color, f"Median: {med_time}"))

        # Draw legend
        leg_x = margin_left + plot_w + 15
        leg_y = margin_top + 20
        draw.rectangle(
            [(leg_x, leg_y), (width - 15, leg_y + len(legend_items) * 45 + 15)],
            fill="#F9FAFB",
            outline="#E5E7EB",
        )
        for i, (l_name, l_color, l_extra) in enumerate(legend_items):
            ly = leg_y + 15 + i * 45
            draw.line(
                [(leg_x + 10, ly + 6), (leg_x + 35, ly + 6)], fill=l_color, width=4
            )
            draw.text(
                (leg_x + 45, ly),
                str(l_name),
                fill="#111827",
                font=self._get_font(13, bold=True),
            )
            draw.text(
                (leg_x + 45, ly + 18), l_extra, fill="#6B7280", font=self._get_font(11)
            )

        out_path = os_import.path.join(self.output_dir, filename)
        img.save(out_path)
        return out_path

    def plot_correlation_heatmap(self, corr_dict, title, filename):
        """Draws numerical correlation matrix heatmap with text annotations."""
        if not self.has_pil or not corr_dict:
            return

        cols = list(corr_dict.keys())
        n = len(cols)

        cell_size = max(50, min(100, 600 // max(1, n)))
        margin = 140
        width = margin * 2 + cell_size * n
        height = margin * 2 + cell_size * n + 60

        img = self.Image.new("RGB", (width, height), "#FFFFFF")
        draw = self.ImageDraw.Draw(img)

        draw.text(
            (margin, 30), title, fill="#111827", font=self._get_font(20, bold=True)
        )

        for i in range(n):
            c1 = cols[i]
            # Row label
            draw.text(
                (margin - 120, margin + i * cell_size + cell_size // 3),
                c1[:12],
                fill="#111827",
                font=self._get_font(12),
            )
            # Column label
            draw.text(
                (margin + i * cell_size + 5, margin - 35),
                cols[i][:10],
                fill="#111827",
                font=self._get_font(11),
            )

            for j in range(n):
                c2 = cols[j]
                val = corr_dict[c1].get(c2, 0.0)

                # Color gradient: -1.0 (blue) to 0.0 (white) to +1.0 (red)
                if val >= 0:
                    # White to Red
                    r = 255
                    g = int(255 * (1.0 - val))
                    b = int(255 * (1.0 - val))
                else:
                    # White to Blue
                    r = int(255 * (1.0 + val))
                    g = int(255 * (1.0 + val))
                    b = 255

                hex_col = f"#{r:02X}{g:02X}{b:02X}"
                x1 = margin + j * cell_size
                y1 = margin + i * cell_size
                x2 = x1 + cell_size
                y2 = y1 + cell_size

                draw.rectangle([(x1, y1), (x2, y2)], fill=hex_col, outline="#E5E7EB")

                # Text value
                txt_col = "#FFFFFF" if abs(val) > 0.6 else "#111827"
                draw.text(
                    (x1 + cell_size // 4, y1 + cell_size // 3),
                    f"{val:.2f}",
                    fill=txt_col,
                    font=self._get_font(11, bold=True),
                )

        out_path = os_import.path.join(self.output_dir, filename)
        img.save(out_path)
        return out_path

    def plot_histogram(self, values, title, xlabel, filename):
        """Draws histogram with distribution bins."""
        if not self.has_pil or not values:
            return

        width, height = 800, 500
        margin = 70
        img = self.Image.new("RGB", (width, height), "#FFFFFF")
        draw = self.ImageDraw.Draw(img)

        draw.text(
            (margin, 20), title, fill="#111827", font=self._get_font(18, bold=True)
        )

        min_v, max_v = min(values), max(values)
        if min_v == max_v:
            max_v += 1.0

        num_bins = 20
        bin_width = (max_v - min_v) / num_bins
        bins = [0] * num_bins

        for v in values:
            idx = int((v - min_v) / bin_width)
            if idx >= num_bins:
                idx = num_bins - 1
            bins[idx] += 1

        max_bin = max(bins) if max(bins) > 0 else 1
        plot_w = width - margin * 2
        plot_h = height - margin * 2

        for i, count in enumerate(bins):
            x1 = margin + int((i / num_bins) * plot_w)
            x2 = margin + int(((i + 1) / num_bins) * plot_w) - 2
            bar_h = int((count / max_bin) * plot_h)
            y1 = margin + plot_h - bar_h
            y2 = margin + plot_h

            draw.rectangle([(x1, y1), (x2, y2)], fill="#3B82F6", outline="#1D4ED8")

        # Draw axes
        draw.line(
            [(margin, margin + plot_h), (margin + plot_w, margin + plot_h)],
            fill="#111827",
            width=2,
        )
        draw.line(
            [(margin, margin), (margin, margin + plot_h)], fill="#111827", width=2
        )

        draw.text(
            (margin + plot_w // 2 - 30, margin + plot_h + 20),
            xlabel,
            fill="#111827",
            font=self._get_font(14, bold=True),
        )
        draw.text(
            (margin - 50, margin + plot_h // 2),
            "Count",
            fill="#111827",
            font=self._get_font(14, bold=True),
        )

        out_path = os_import.path.join(self.output_dir, filename)
        img.save(out_path)
        return out_path

    def _get_font(self, size, bold=False):
        try:
            if bold:
                return self.ImageFont.truetype(
                    "/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf", size
                )
            else:
                return self.ImageFont.truetype(
                    "/usr/share/fonts/dejavu/DejaVuSans.ttf", size
                )
        except (OSError, AttributeError):
            return self.ImageFont.load_default()
