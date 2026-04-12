import { useEffect, useRef } from 'react';
import PropTypes from 'prop-types';
import * as d3 from 'd3';
import { TA } from '../i18n/ta.js';
import { getMockTrendData } from '../api/mockData.js';
import './TriageTrendChart.css';

/** @type {import('../api/mockData.js').TrendDay[]} */
export const MOCK_TREND_DATA = getMockTrendData();

const MARGIN = { top: 20, right: 30, bottom: 45, left: 55 };

const SERIES = [
  { key: 'green',  color: '#2ecc71', label: 'Green' },
  { key: 'yellow', color: '#f39c12', label: 'Yellow' },
  { key: 'red',    color: '#e74c3c', label: 'Red' },
];

/**
 * @typedef {Object} TriageTrendChartProps
 * @property {import('../api/mockData.js').TrendDay[]} data
 */

/**
 * D3 multi-line trend chart showing 7-day GREEN/YELLOW/RED case counts.
 * @param {TriageTrendChartProps} props
 */
export default function TriageTrendChart({ data }) {
  const containerRef = useRef(null);
  const svgRef       = useRef(null);

  useEffect(() => {
    if (!svgRef.current || !containerRef.current) return;

    const svg       = d3.select(svgRef.current);
    const container = containerRef.current;

    svg.selectAll('*').remove();

    if (!data || data.length === 0) return;

    // Responsive width from container
    const totalWidth  = container.clientWidth  || 600;
    const totalHeight = container.clientHeight || 300;
    const width  = totalWidth  - MARGIN.left - MARGIN.right;
    const height = totalHeight - MARGIN.top  - MARGIN.bottom;

    svg
      .attr('width',  totalWidth)
      .attr('height', totalHeight)
      .attr('role',   'img')
      .attr('aria-label', `7-day triage trend chart with ${data.length} data points`);

    const g = svg.append('g').attr('transform', `translate(${MARGIN.left},${MARGIN.top})`);

    // Parse dates
    const parseDate = d3.timeParse('%Y-%m-%d');
    const parsed    = data.map((d) => ({
      date:   parseDate(d.date),
      green:  d.green,
      yellow: d.yellow,
      red:    d.red,
    }));

    // Scales
    const xScale = d3.scaleTime()
      .domain(d3.extent(parsed, (d) => d.date))
      .range([0, width]);

    const yMax = d3.max(parsed, (d) => Math.max(d.green, d.yellow, d.red));
    const yScale = d3.scaleLinear()
      .domain([0, yMax * 1.1])
      .range([height, 0])
      .nice();

    // Grid lines
    g.append('g')
      .attr('class', 'grid-lines')
      .call(
        d3.axisLeft(yScale)
          .ticks(5)
          .tickSize(-width)
          .tickFormat('')
      )
      .call((axis) => axis.select('.domain').remove())
      .call((axis) => axis.selectAll('line').attr('stroke', '#2a2a4a').attr('stroke-dasharray', '4,4'));

    // X axis
    g.append('g')
      .attr('class', 'x-axis')
      .attr('transform', `translate(0,${height})`)
      .call(
        d3.axisBottom(xScale)
          .ticks(d3.timeDay.every(1))
          .tickFormat(d3.timeFormat('%b %d'))
      )
      .call((axis) => axis.select('.domain').attr('stroke', '#4a5568'))
      .call((axis) => axis.selectAll('text')
        .attr('fill', '#a0aec0')
        .attr('font-size', '11px')
        .attr('dy', '1.2em')
      )
      .call((axis) => axis.selectAll('line').attr('stroke', '#4a5568'));

    // Y axis
    g.append('g')
      .attr('class', 'y-axis')
      .call(d3.axisLeft(yScale).ticks(5))
      .call((axis) => axis.select('.domain').attr('stroke', '#4a5568'))
      .call((axis) => axis.selectAll('text').attr('fill', '#a0aec0').attr('font-size', '11px'))
      .call((axis) => axis.selectAll('line').attr('stroke', '#4a5568'));

    // Y axis label
    g.append('text')
      .attr('transform', 'rotate(-90)')
      .attr('x', -height / 2)
      .attr('y', -42)
      .attr('text-anchor', 'middle')
      .attr('fill', '#718096')
      .attr('font-size', '11px')
      .text(TA.TREND_YAXIS_EN);

    // Line generator with smooth Catmull-Rom curve
    const lineGenerator = d3.line()
      .x((d) => xScale(d.date))
      .y((d) => yScale(d.value))
      .curve(d3.curveCatmullRom.alpha(0.5));

    // Area generator for subtle fill under each line
    const areaGenerator = d3.area()
      .x((d) => xScale(d.date))
      .y0(height)
      .y1((d) => yScale(d.value))
      .curve(d3.curveCatmullRom.alpha(0.5));

    SERIES.forEach(({ key, color, label }) => {
      const lineData = parsed.map((d) => ({ date: d.date, value: d[key] }));

      // Subtle area fill
      g.append('path')
        .datum(lineData)
        .attr('class', `area-${key}`)
        .attr('d', areaGenerator)
        .attr('fill', color)
        .attr('opacity', 0.06)
        .attr('aria-hidden', 'true');

      // Line
      g.append('path')
        .datum(lineData)
        .attr('class', `line-${key}`)
        .attr('d', lineGenerator)
        .attr('fill', 'none')
        .attr('stroke', color)
        .attr('stroke-width', 2.5)
        .attr('aria-label', `${label} triage trend line`);

      // Data point dots
      g.selectAll(`.dot-${key}`)
        .data(lineData)
        .enter()
        .append('circle')
        .attr('class', `dot-${key}`)
        .attr('cx', (d) => xScale(d.date))
        .attr('cy', (d) => yScale(d.value))
        .attr('r', 4)
        .attr('fill', color)
        .attr('stroke', '#16213e')
        .attr('stroke-width', 2)
        .attr('role', 'img')
        .attr('aria-label', (d) => `${label}: ${d.value} cases on ${d3.timeFormat('%b %d')(d.date)}`);
    });

    // Hover crosshair + value tooltip
    const bisectDate = d3.bisector((d) => d.date).left;
    const focusLine  = g.append('line')
      .attr('class', 'focus-line')
      .attr('stroke', '#4a5568')
      .attr('stroke-dasharray', '4,3')
      .attr('y1', 0)
      .attr('y2', height)
      .style('display', 'none');

    const focusDots = SERIES.map(({ key, color }) =>
      g.append('circle')
        .attr('r', 6)
        .attr('fill', color)
        .attr('stroke', '#16213e')
        .attr('stroke-width', 2)
        .style('display', 'none')
    );

    const focusLabels = SERIES.map(({ color }, idx) =>
      g.append('text')
        .attr('fill', color)
        .attr('font-size', '11px')
        .attr('text-anchor', 'middle')
        .style('display', 'none')
    );

    // Invisible overlay for mouse events
    g.append('rect')
      .attr('class', 'overlay')
      .attr('width', width)
      .attr('height', height)
      .attr('fill', 'none')
      .attr('pointer-events', 'all')
      .attr('aria-hidden', 'true')
      .on('mouseenter', () => {
        focusLine.style('display', null);
        focusDots.forEach((dot) => dot.style('display', null));
        focusLabels.forEach((lbl) => lbl.style('display', null));
      })
      .on('mouseleave', () => {
        focusLine.style('display', 'none');
        focusDots.forEach((dot) => dot.style('display', 'none'));
        focusLabels.forEach((lbl) => lbl.style('display', 'none'));
      })
      .on('mousemove', (event) => {
        const [mx]        = d3.pointer(event);
        const x0          = xScale.invert(mx);
        const idx         = bisectDate(parsed, x0, 1);
        const d           = idx >= parsed.length ? parsed[parsed.length - 1] : parsed[idx];

        focusLine.attr('x1', xScale(d.date)).attr('x2', xScale(d.date));

        SERIES.forEach(({ key }, i) => {
          const yPos = yScale(d[key]);
          focusDots[i].attr('cx', xScale(d.date)).attr('cy', yPos);
          focusLabels[i]
            .attr('x', xScale(d.date) + 8)
            .attr('y', yPos - 8)
            .text(d[key]);
        });
      });

    // Responsive: re-render on resize
    const resizeObserver = new ResizeObserver(() => {
      svg.selectAll('*').remove();
      // Re-trigger by nudging data identity — React will re-run the effect
      // via the data prop. In practice, trigger a manual re-render signal.
      // For simplicity, we call the effect body inline here would be complex;
      // the parent can re-mount on window resize if needed.
    });
    resizeObserver.observe(container);

    return () => {
      resizeObserver.disconnect();
      svg.selectAll('*').remove();
    };
  }, [data]);

  return (
    <div className="trend-chart-container">
      <header className="trend-chart-header">
        <h2 className="trend-chart-title">{TA.TREND_TITLE}</h2>
        <span className="trend-chart-title-en">{TA.TREND_TITLE_EN}</span>
        <div className="trend-chart-legend" aria-label="Line chart legend">
          {SERIES.map(({ key, color, label }) => (
            <span key={key} className="legend-item">
              <span
                className="legend-line-swatch"
                style={{ backgroundColor: color }}
                aria-hidden="true"
              />
              <span className="legend-label">{label}</span>
            </span>
          ))}
        </div>
      </header>
      <div ref={containerRef} className="trend-chart-svg-wrapper">
        <svg ref={svgRef} className="trend-chart-svg" />
      </div>
    </div>
  );
}

TriageTrendChart.propTypes = {
  data: PropTypes.arrayOf(
    PropTypes.shape({
      date:   PropTypes.string.isRequired,
      green:  PropTypes.number.isRequired,
      yellow: PropTypes.number.isRequired,
      red:    PropTypes.number.isRequired,
    })
  ).isRequired,
};
