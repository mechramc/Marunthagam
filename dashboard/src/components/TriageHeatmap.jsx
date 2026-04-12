import { useEffect, useRef } from 'react';
import PropTypes from 'prop-types';
import * as d3 from 'd3';
import { TA } from '../i18n/ta.js';
import { getMockHeatmapData } from '../api/mockData.js';
import './TriageHeatmap.css';

/** @type {import('../api/mockData.js').HeatmapCell[]} */
export const MOCK_HEATMAP_DATA = getMockHeatmapData();

const LEVEL_COLORS = {
  green:  '#2ecc71',
  yellow: '#f39c12',
  red:    '#e74c3c',
  empty:  '#2a2a4a',
};

const CELL_SIZE   = 80;  // px per cell
const CELL_GAP    = 6;   // px gap between cells
const CELL_STRIDE = CELL_SIZE + CELL_GAP;
const COLS        = 5;
const MARGIN      = { top: 10, right: 20, bottom: 10, left: 20 };

/**
 * Determines the dominant triage level for a cell.
 * @param {import('../api/mockData.js').HeatmapCell} cell
 * @returns {'green'|'yellow'|'red'|'empty'}
 */
function dominantLevel(cell) {
  const { green_count: g, yellow_count: y, red_count: r } = cell;
  if (g === 0 && y === 0 && r === 0) return 'empty';
  // Red always wins for display if any present
  if (r > 0) return 'red';
  if (y > g) return 'yellow';
  return 'green';
}

/**
 * Formats an ISO timestamp to a readable date/time string.
 * @param {string} isoString
 * @returns {string}
 */
function formatTimestamp(isoString) {
  return new Date(isoString).toLocaleString('en-IN', {
    month:  'short',
    day:    'numeric',
    hour:   '2-digit',
    minute: '2-digit',
  });
}

/**
 * @typedef {Object} TriageHeatmapProps
 * @property {import('../api/mockData.js').HeatmapCell[]} data
 */

/**
 * D3-rendered geohash grid heatmap.
 * Each cell is colored by dominant triage level; opacity scales with total case count.
 * @param {TriageHeatmapProps} props
 */
export default function TriageHeatmap({ data }) {
  const svgRef     = useRef(null);
  const tooltipRef = useRef(null);

  useEffect(() => {
    if (!svgRef.current || !tooltipRef.current) return;

    const svg     = d3.select(svgRef.current);
    const tooltip = d3.select(tooltipRef.current);

    // Clear previous render
    svg.selectAll('*').remove();

    if (!data || data.length === 0) {
      svg
        .append('text')
        .attr('x', '50%')
        .attr('y', '50%')
        .attr('text-anchor', 'middle')
        .attr('fill', '#718096')
        .text(TA.HEATMAP_NO_DATA);
      return;
    }

    const rows      = Math.ceil(data.length / COLS);
    const svgWidth  = COLS * CELL_STRIDE - CELL_GAP + MARGIN.left + MARGIN.right;
    const svgHeight = rows * CELL_STRIDE - CELL_GAP + MARGIN.top  + MARGIN.bottom;

    svg
      .attr('width',  svgWidth)
      .attr('height', svgHeight)
      .attr('role',   'img')
      .attr('aria-label', `Triage heatmap showing ${data.length} geohash cells`);

    const g = svg
      .append('g')
      .attr('transform', `translate(${MARGIN.left},${MARGIN.top})`);

    // Opacity scale: map total count to [0.3, 1.0]
    const maxTotal = d3.max(data, (d) => d.green_count + d.yellow_count + d.red_count) || 1;
    const opacityScale = d3.scaleLinear()
      .domain([0, maxTotal])
      .range([0.3, 1.0])
      .clamp(true);

    // Render cells
    data.forEach((cell, i) => {
      const col   = i % COLS;
      const row   = Math.floor(i / COLS);
      const x     = col * CELL_STRIDE;
      const y     = row * CELL_STRIDE;
      const level = dominantLevel(cell);
      const total = cell.green_count + cell.yellow_count + cell.red_count;

      const cellGroup = g
        .append('g')
        .attr('transform', `translate(${x},${y})`)
        .attr('role', 'button')
        .attr('tabindex', '0')
        .attr('aria-label',
          `${TA.HEATMAP_TOOLTIP_GEOHASH} ${cell.geohash}: ` +
          `Green ${cell.green_count}, Yellow ${cell.yellow_count}, Red ${cell.red_count}`
        );

      // Background rect
      cellGroup
        .append('rect')
        .attr('width',  CELL_SIZE)
        .attr('height', CELL_SIZE)
        .attr('rx', 6)
        .attr('ry', 6)
        .attr('fill',    LEVEL_COLORS[level])
        .attr('opacity', opacityScale(total));

      // Geohash label (truncated to last 3 chars to fit)
      cellGroup
        .append('text')
        .attr('x', CELL_SIZE / 2)
        .attr('y', CELL_SIZE / 2 - 8)
        .attr('text-anchor', 'middle')
        .attr('dominant-baseline', 'middle')
        .attr('fill', '#fff')
        .attr('font-size', '11px')
        .attr('font-family', 'monospace')
        .attr('opacity', 0.9)
        .text(cell.geohash.slice(-4));

      // Total count
      cellGroup
        .append('text')
        .attr('x', CELL_SIZE / 2)
        .attr('y', CELL_SIZE / 2 + 10)
        .attr('text-anchor', 'middle')
        .attr('dominant-baseline', 'middle')
        .attr('fill', '#fff')
        .attr('font-size', '13px')
        .attr('font-weight', '700')
        .text(total);

      // Red indicator dot (bottom-right) when red > 0
      if (cell.red_count > 0) {
        cellGroup
          .append('circle')
          .attr('cx', CELL_SIZE - 10)
          .attr('cy', 10)
          .attr('r',  5)
          .attr('fill', '#e74c3c')
          .attr('stroke', '#fff')
          .attr('stroke-width', 1.5);
      }

      // Tooltip on hover / focus
      const showTooltip = (event) => {
        const [px, py] = [event.pageX, event.pageY];
        tooltip
          .style('display', 'block')
          .style('left',    `${px + 12}px`)
          .style('top',     `${py - 28}px`)
          .html(
            `<strong>${TA.HEATMAP_TOOLTIP_GEOHASH}:</strong> ${cell.geohash}<br>` +
            `<span style="color:${LEVEL_COLORS.green}">G</span>: ${cell.green_count} &nbsp;` +
            `<span style="color:${LEVEL_COLORS.yellow}">Y</span>: ${cell.yellow_count} &nbsp;` +
            `<span style="color:${LEVEL_COLORS.red}">R</span>: ${cell.red_count}<br>` +
            `<small>${TA.HEATMAP_TOOLTIP_LAST_UPDATED}: ${formatTimestamp(cell.last_updated)}</small>`
          );
      };

      const hideTooltip = () => {
        tooltip.style('display', 'none');
      };

      cellGroup
        .on('mouseenter', showTooltip)
        .on('mousemove',  showTooltip)
        .on('mouseleave', hideTooltip)
        .on('focus',      showTooltip)
        .on('blur',       hideTooltip);
    });

    // Cleanup: remove tooltip on unmount
    return () => {
      tooltip.style('display', 'none');
      svg.selectAll('*').remove();
    };
  }, [data]);

  return (
    <div className="heatmap-container">
      <header className="heatmap-header">
        <h2 className="heatmap-title">{TA.HEATMAP_TITLE}</h2>
        <span className="heatmap-title-en">{TA.HEATMAP_TITLE_EN}</span>
        <div className="heatmap-legend" aria-label="Color legend">
          {Object.entries(LEVEL_COLORS)
            .filter(([key]) => key !== 'empty')
            .map(([level, color]) => (
              <span key={level} className="legend-item">
                <span
                  className="legend-swatch"
                  style={{ backgroundColor: color }}
                  aria-hidden="true"
                />
                <span className="legend-label">{level.charAt(0).toUpperCase() + level.slice(1)}</span>
              </span>
            ))}
        </div>
      </header>
      <div className="heatmap-svg-wrapper">
        <svg ref={svgRef} className="heatmap-svg" />
        {/* Tooltip rendered in DOM to allow full CSS hover styling */}
        <div ref={tooltipRef} className="heatmap-tooltip" role="tooltip" aria-live="polite" />
      </div>
    </div>
  );
}

TriageHeatmap.propTypes = {
  data: PropTypes.arrayOf(
    PropTypes.shape({
      geohash:      PropTypes.string.isRequired,
      green_count:  PropTypes.number.isRequired,
      yellow_count: PropTypes.number.isRequired,
      red_count:    PropTypes.number.isRequired,
      last_updated: PropTypes.string.isRequired,
    })
  ).isRequired,
};
