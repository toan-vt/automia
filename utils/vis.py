"""
MIA Experiments Dashboard Generator

Generates an interactive HTML dashboard displaying experiment results with:
- Tree/hierarchy view of parent/child relationships
- Sortable/filterable table
- Links to runtime code files
- Expandable details for each experiment

Usage:
    python -m mia_agents_dual.vis --table-name dev --output-dir agent-mia-results
"""

import argparse
import json
import html
import os
from pathlib import Path
from typing import Optional
import psycopg2


def parse_args():
    parser = argparse.ArgumentParser(description="Generate MIA experiments dashboard")
    parser.add_argument("--table-name", type=str, default="dev", help="Database table name (or subdirectory name for file backend)")
    parser.add_argument("--backend", type=str, default="file", choices=["postgres", "file"], help="Storage backend")
    parser.add_argument("--data-dir", type=str, default=None, help="Parent directory containing table subdirectories (file backend only)")
    parser.add_argument("--db-name", type=str, default="mia", help="Database name (postgres backend only)")
    parser.add_argument("--db-user", type=str, default="user", help="Database user (postgres backend only)")
    parser.add_argument("--db-password", type=str, default="", help="Database password (postgres backend only)")
    parser.add_argument("--output-dir", type=str, default="agent-mia-results", help="Output directory (dashboard.html will be saved here)")
    return parser.parse_args()


def fetch_experiments(db_name: str, db_user: str, db_password: str, table_name: str) -> list[dict]:
    """Fetch all experiments from a PostgreSQL database."""
    conn = psycopg2.connect(dbname=db_name, user=db_user, password=db_password)
    cursor = conn.cursor()
    cursor.execute(f"""
        SELECT id, idea, design_justification, implementation, analysis_summary,
               auc_score, tpr_1_score, tpr_5_score, combined_score, parent_id
        FROM {table_name}
        ORDER BY id
    """)
    cols = [desc[0] for desc in cursor.description]
    experiments = [dict(zip(cols, row)) for row in cursor.fetchall()]
    conn.close()
    return experiments


def fetch_experiments_from_files(data_dir: str, table_name: str) -> list[dict]:
    """Fetch all experiments from the file backend (one JSON per row)."""
    dir_path = Path(data_dir) / table_name
    if not dir_path.exists():
        raise FileNotFoundError(f"Directory not found: {dir_path}")
    records = []
    for p in sorted(dir_path.glob("*.json"), key=lambda x: int(x.stem)):
        try:
            record = json.loads(p.read_text())
            # Drop embedding columns — not needed for visualization
            record = {k: v for k, v in record.items() if not k.endswith("_embedding")}
            records.append(record)
        except Exception:
            pass
    return records


def find_runtime_files(output_dir: str) -> dict[int, str]:
    """Scan runtime directory and map by creation order to experiment IDs."""
    runtime_dir = Path(output_dir) / "runtime"
    if not runtime_dir.exists():
        return {}

    # Get all runtime subdirectories sorted by name (contains timestamp)
    subdirs = sorted(runtime_dir.iterdir(), key=lambda p: p.name)

    # Map index to path (heuristic: experiments are created in order)
    # This is imperfect but provides a reasonable mapping
    return {i + 1: str(subdir / "mia_run.py") for i, subdir in enumerate(subdirs) if subdir.is_dir()}


def build_tree(experiments: list[dict]) -> dict:
    """Build tree structure from experiments based on parent_id."""
    by_id = {exp["id"]: exp for exp in experiments}
    roots = []
    children_map = {}

    for exp in experiments:
        parent_id = exp.get("parent_id", -1)
        if parent_id == -1 or parent_id not in by_id:
            roots.append(exp)
        else:
            if parent_id not in children_map:
                children_map[parent_id] = []
            children_map[parent_id].append(exp)

    return {"roots": roots, "children_map": children_map}


def escape(text: Optional[str]) -> str:
    """HTML escape text, handling None."""
    if text is None:
        return ""
    return html.escape(str(text))


def generate_html(experiments: list[dict], runtime_files: dict[int, str], output_dir: str) -> str:
    """Generate the complete HTML dashboard."""
    tree = build_tree(experiments)

    # Convert experiments to JSON for JavaScript
    experiments_json = json.dumps([
        {
            "id": exp["id"],
            "idea": exp.get("idea", ""),
            "design_justification": exp.get("design_justification", ""),
            "implementation": exp.get("implementation", ""),
            "analysis_summary": exp.get("analysis_summary", ""),
            "auc_score": exp.get("auc_score"),
            "tpr_1_score": exp.get("tpr_1_score"),
            "tpr_5_score": exp.get("tpr_5_score"),
            "combined_score": exp.get("combined_score"),
            "parent_id": exp.get("parent_id", -1),
            "code_path": runtime_files.get(exp["id"], "")
        }
        for exp in experiments
    ])

    html_content = f'''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>MIA Experiments Dashboard</title>
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.9.0/styles/github-dark.min.css">
    <script src="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.9.0/highlight.min.js"></script>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.9.0/languages/python.min.js"></script>
    <style>
        * {{
            box-sizing: border-box;
        }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, sans-serif;
            margin: 0;
            padding: 20px;
            background: #f5f5f5;
            color: #333;
        }}
        h1 {{
            margin-bottom: 20px;
            color: #1a1a2e;
        }}
        .controls {{
            display: flex;
            gap: 15px;
            margin-bottom: 20px;
            flex-wrap: wrap;
            align-items: center;
        }}
        .controls input, .controls select {{
            padding: 8px 12px;
            border: 1px solid #ddd;
            border-radius: 4px;
            font-size: 14px;
        }}
        .controls input[type="text"] {{
            width: 300px;
        }}
        .controls label {{
            font-weight: 500;
            margin-right: 5px;
        }}
        .view-toggle {{
            display: flex;
            gap: 5px;
        }}
        .view-toggle button {{
            padding: 8px 16px;
            border: 1px solid #ddd;
            background: white;
            cursor: pointer;
            border-radius: 4px;
        }}
        .view-toggle button.active {{
            background: #4a90d9;
            color: white;
            border-color: #4a90d9;
        }}
        .stats {{
            background: white;
            padding: 15px 20px;
            border-radius: 8px;
            margin-bottom: 20px;
            display: flex;
            gap: 30px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        }}
        .stat-item {{
            text-align: center;
        }}
        .stat-value {{
            font-size: 24px;
            font-weight: bold;
            color: #4a90d9;
        }}
        .stat-label {{
            font-size: 12px;
            color: #666;
            text-transform: uppercase;
        }}
        .tree-view, .table-view {{
            background: white;
            border-radius: 8px;
            padding: 20px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        }}
        .hidden {{
            display: none !important;
        }}
        /* Tree View Styles */
        .tree-node {{
            margin-left: 20px;
            border-left: 2px solid #e0e0e0;
            padding-left: 15px;
            margin-bottom: 10px;
        }}
        .tree-node.root {{
            margin-left: 0;
            border-left: none;
            padding-left: 0;
        }}
        .experiment-card {{
            background: #fafafa;
            border: 1px solid #e0e0e0;
            border-radius: 6px;
            padding: 15px;
            margin-bottom: 10px;
        }}
        .experiment-card:hover {{
            border-color: #4a90d9;
        }}
        .card-header {{
            display: flex;
            justify-content: space-between;
            align-items: flex-start;
            cursor: pointer;
        }}
        .card-title {{
            font-weight: 600;
            color: #1a1a2e;
            margin-bottom: 5px;
        }}
        .card-id {{
            font-size: 12px;
            color: #888;
            background: #eee;
            padding: 2px 6px;
            border-radius: 3px;
        }}
        .card-scores {{
            display: flex;
            gap: 10px;
            margin-top: 8px;
        }}
        .score-badge {{
            font-size: 12px;
            padding: 3px 8px;
            border-radius: 4px;
            background: #e8f4fd;
            color: #2c5aa0;
        }}
        .score-badge.auc {{
            background: #e8f4fd;
            color: #2c5aa0;
        }}
        .score-badge.tpr1 {{
            background: #e8fde8;
            color: #2c7a2c;
        }}
        .score-badge.tpr5 {{
            background: #fdf4e8;
            color: #9a6c28;
        }}
        .card-details {{
            margin-top: 15px;
            padding-top: 15px;
            border-top: 1px solid #e0e0e0;
            display: none;
        }}
        .card-details.expanded {{
            display: block;
        }}
        .detail-section {{
            margin-bottom: 15px;
        }}
        .detail-label {{
            font-weight: 600;
            font-size: 13px;
            color: #555;
            margin-bottom: 5px;
        }}
        .detail-content {{
            font-size: 14px;
            line-height: 1.5;
            color: #333;
            white-space: pre-wrap;
        }}
        .code-block {{
            border-radius: 4px;
            font-family: 'Fira Code', 'Consolas', monospace;
            font-size: 13px;
            max-height: 400px;
            overflow: auto;
            margin: 0;
        }}
        .code-block code {{
            font-family: inherit;
            font-size: inherit;
        }}
        .code-link {{
            display: inline-block;
            margin-top: 5px;
            color: #4a90d9;
            text-decoration: none;
            font-size: 13px;
        }}
        .code-link:hover {{
            text-decoration: underline;
        }}
        .toggle-icon {{
            font-size: 18px;
            color: #888;
            transition: transform 0.2s;
        }}
        .toggle-icon.expanded {{
            transform: rotate(90deg);
        }}
        .children-container {{
            margin-top: 10px;
        }}
        .collapse-btn {{
            background: none;
            border: none;
            color: #4a90d9;
            cursor: pointer;
            font-size: 12px;
            padding: 0;
            margin-top: 5px;
        }}
        /* Table View Styles */
        table {{
            width: 100%;
            border-collapse: collapse;
        }}
        th, td {{
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #e0e0e0;
        }}
        th {{
            background: #f8f9fa;
            font-weight: 600;
            cursor: pointer;
            user-select: none;
        }}
        th:hover {{
            background: #e9ecef;
        }}
        th .sort-icon {{
            margin-left: 5px;
            opacity: 0.3;
        }}
        th.sorted .sort-icon {{
            opacity: 1;
        }}
        tr:hover {{
            background: #f8f9fa;
        }}
        .idea-cell {{
            max-width: 300px;
            overflow: hidden;
            text-overflow: ellipsis;
            white-space: nowrap;
        }}
        .expand-row {{
            cursor: pointer;
            color: #4a90d9;
        }}
        .expanded-row {{
            background: #f8f9fa;
        }}
        .expanded-row td {{
            padding: 20px;
        }}
        .no-results {{
            text-align: center;
            padding: 40px;
            color: #888;
        }}
    </style>
</head>
<body>
    <h1>MIA Experiments Dashboard</h1>

    <div class="stats" id="stats">
        <div class="stat-item">
            <div class="stat-value" id="total-count">0</div>
            <div class="stat-label">Total Experiments</div>
        </div>
        <div class="stat-item">
            <div class="stat-value" id="best-auc">0</div>
            <div class="stat-label">Best AUC</div>
        </div>
        <div class="stat-item">
            <div class="stat-value" id="best-tpr1">0</div>
            <div class="stat-label">Best TPR@1%</div>
        </div>
        <div class="stat-item">
            <div class="stat-value" id="best-tpr5">0</div>
            <div class="stat-label">Best TPR@5%</div>
        </div>
        <div class="stat-item">
            <div class="stat-value" id="root-count">0</div>
            <div class="stat-label">Root Experiments</div>
        </div>
    </div>

    <div class="controls">
        <div>
            <label>Search:</label>
            <input type="text" id="search" placeholder="Search ideas, justifications...">
        </div>
        <div>
            <label>Sort by:</label>
            <select id="sort-by">
                <option value="id">ID</option>
                <option value="auc_score">AUC Score</option>
                <option value="tpr_1_score">TPR@1%</option>
                <option value="tpr_5_score">TPR@5%</option>
                <option value="combined_score">Combined Score</option>
            </select>
        </div>
        <div>
            <label>Order:</label>
            <select id="sort-order">
                <option value="desc">Descending</option>
                <option value="asc">Ascending</option>
            </select>
        </div>
        <div>
            <label>Min AUC:</label>
            <input type="number" id="min-auc" step="0.01" min="0" max="1" placeholder="0.0" style="width: 80px;">
        </div>
        <div class="view-toggle">
            <button id="tree-btn" class="active">Tree View</button>
            <button id="table-btn">Table View</button>
        </div>
    </div>

    <div class="tree-view" id="tree-view"></div>
    <div class="table-view hidden" id="table-view"></div>

    <script>
        const experiments = {experiments_json};
        const outputDir = {json.dumps(output_dir)};

        // Build parent-children map
        const childrenMap = {{}};
        const byId = {{}};
        experiments.forEach(exp => {{
            byId[exp.id] = exp;
            const parentId = exp.parent_id;
            if (parentId !== -1 && byId[parentId]) {{
                if (!childrenMap[parentId]) childrenMap[parentId] = [];
                childrenMap[parentId].push(exp);
            }}
        }});

        // Re-process after byId is populated
        experiments.forEach(exp => {{
            const parentId = exp.parent_id;
            if (parentId !== -1 && byId[parentId]) {{
                if (!childrenMap[parentId]) childrenMap[parentId] = [];
                if (!childrenMap[parentId].includes(exp)) {{
                    childrenMap[parentId].push(exp);
                }}
            }}
        }});

        function getRoots(exps) {{
            return exps.filter(exp => exp.parent_id === -1 || !byId[exp.parent_id]);
        }}

        function formatScore(score) {{
            if (score === null || score === undefined) return 'N/A';
            return score.toFixed(4);
        }}

        function escapeHtml(text) {{
            if (!text) return '';
            const div = document.createElement('div');
            div.textContent = text;
            return div.innerHTML;
        }}

        function createExperimentCard(exp, isRoot = false) {{
            const children = childrenMap[exp.id] || [];
            const hasChildren = children.length > 0;

            return `
                <div class="tree-node ${{isRoot ? 'root' : ''}}" data-id="${{exp.id}}">
                    <div class="experiment-card">
                        <div class="card-header" onclick="toggleCard(${{exp.id}})">
                            <div>
                                <span class="card-id">#${{exp.id}}${{exp.parent_id !== -1 ? ' (child of #' + exp.parent_id + ')' : ''}}</span>
                                <div class="card-title">${{escapeHtml(exp.idea?.substring(0, 150))}}</div>
                                <div class="card-scores">
                                    <span class="score-badge auc">AUC: ${{formatScore(exp.auc_score)}}</span>
                                    <span class="score-badge tpr1">TPR@1%: ${{formatScore(exp.tpr_1_score)}}</span>
                                    <span class="score-badge tpr5">TPR@5%: ${{formatScore(exp.tpr_5_score)}}</span>
                                </div>
                            </div>
                            <span class="toggle-icon" id="icon-${{exp.id}}">▶</span>
                        </div>
                        <div class="card-details" id="details-${{exp.id}}">
                            <div class="detail-section">
                                <div class="detail-label">Idea</div>
                                <div class="detail-content">${{escapeHtml(exp.idea)}}</div>
                            </div>
                            <div class="detail-section">
                                <div class="detail-label">Design Justification</div>
                                <div class="detail-content">${{escapeHtml(exp.design_justification)}}</div>
                            </div>
                            <div class="detail-section">
                                <div class="detail-label">Analysis Summary</div>
                                <div class="detail-content">${{escapeHtml(exp.analysis_summary)}}</div>
                            </div>
                            <div class="detail-section">
                                <div class="detail-label">Implementation</div>
                                <pre class="code-block"><code class="language-python">${{escapeHtml(exp.implementation)}}</code></pre>
                                ${{exp.code_path ? `<a class="code-link" href="file://${{exp.code_path}}" target="_blank">📁 ${{exp.code_path}}</a>` : ''}}
                            </div>
                            <div class="detail-section">
                                <div class="detail-label">Scores</div>
                                <div class="detail-content">
                                    AUC: ${{formatScore(exp.auc_score)}} |
                                    TPR@1%: ${{formatScore(exp.tpr_1_score)}} |
                                    TPR@5%: ${{formatScore(exp.tpr_5_score)}} |
                                    Combined: ${{formatScore(exp.combined_score)}}
                                </div>
                            </div>
                        </div>
                    </div>
                    ${{hasChildren ? `
                        <button class="collapse-btn" onclick="toggleChildren(${{exp.id}})">
                            <span id="children-toggle-${{exp.id}}">▼</span> ${{children.length}} child experiment(s)
                        </button>
                        <div class="children-container" id="children-${{exp.id}}">
                            ${{children.map(child => createExperimentCard(child, false)).join('')}}
                        </div>
                    ` : ''}}
                </div>
            `;
        }}

        function toggleCard(id) {{
            const details = document.getElementById(`details-${{id}}`);
            const icon = document.getElementById(`icon-${{id}}`);
            if (details.classList.contains('expanded')) {{
                details.classList.remove('expanded');
                icon.classList.remove('expanded');
            }} else {{
                details.classList.add('expanded');
                icon.classList.add('expanded');
                highlightCode();
            }}
        }}

        function toggleChildren(id) {{
            const container = document.getElementById(`children-${{id}}`);
            const toggle = document.getElementById(`children-toggle-${{id}}`);
            if (container.classList.contains('hidden')) {{
                container.classList.remove('hidden');
                toggle.textContent = '▼';
            }} else {{
                container.classList.add('hidden');
                toggle.textContent = '▶';
            }}
        }}

        function filterAndSort() {{
            const searchTerm = document.getElementById('search').value.toLowerCase();
            const sortBy = document.getElementById('sort-by').value;
            const sortOrder = document.getElementById('sort-order').value;
            const minAuc = parseFloat(document.getElementById('min-auc').value) || 0;

            let filtered = experiments.filter(exp => {{
                const matchesSearch = !searchTerm ||
                    (exp.idea && exp.idea.toLowerCase().includes(searchTerm)) ||
                    (exp.design_justification && exp.design_justification.toLowerCase().includes(searchTerm)) ||
                    (exp.analysis_summary && exp.analysis_summary.toLowerCase().includes(searchTerm));
                const matchesAuc = !exp.auc_score || exp.auc_score >= minAuc;
                return matchesSearch && matchesAuc;
            }});

            filtered.sort((a, b) => {{
                let aVal = a[sortBy] ?? -Infinity;
                let bVal = b[sortBy] ?? -Infinity;
                if (sortOrder === 'asc') return aVal - bVal;
                return bVal - aVal;
            }});

            return filtered;
        }}

        function renderTreeView() {{
            const filtered = filterAndSort();
            const roots = getRoots(filtered);

            if (roots.length === 0) {{
                document.getElementById('tree-view').innerHTML = '<div class="no-results">No experiments match your filters</div>';
                return;
            }}

            document.getElementById('tree-view').innerHTML = roots.map(exp => createExperimentCard(exp, true)).join('');
        }}

        function renderTableView() {{
            const filtered = filterAndSort();

            if (filtered.length === 0) {{
                document.getElementById('table-view').innerHTML = '<div class="no-results">No experiments match your filters</div>';
                return;
            }}

            const rows = filtered.map(exp => `
                <tr onclick="toggleTableRow(${{exp.id}})">
                    <td>${{exp.id}}</td>
                    <td>${{exp.parent_id === -1 ? '-' : exp.parent_id}}</td>
                    <td class="idea-cell" title="${{escapeHtml(exp.idea)}}">${{escapeHtml(exp.idea?.substring(0, 80))}}</td>
                    <td>${{formatScore(exp.auc_score)}}</td>
                    <td>${{formatScore(exp.tpr_1_score)}}</td>
                    <td>${{formatScore(exp.tpr_5_score)}}</td>
                    <td class="expand-row">▶</td>
                </tr>
                <tr class="expanded-row hidden" id="table-details-${{exp.id}}">
                    <td colspan="7">
                        <div class="detail-section">
                            <div class="detail-label">Idea</div>
                            <div class="detail-content">${{escapeHtml(exp.idea)}}</div>
                        </div>
                        <div class="detail-section">
                            <div class="detail-label">Design Justification</div>
                            <div class="detail-content">${{escapeHtml(exp.design_justification)}}</div>
                        </div>
                        <div class="detail-section">
                            <div class="detail-label">Analysis Summary</div>
                            <div class="detail-content">${{escapeHtml(exp.analysis_summary)}}</div>
                        </div>
                        <div class="detail-section">
                            <div class="detail-label">Implementation</div>
                            <pre class="code-block"><code class="language-python">${{escapeHtml(exp.implementation)}}</code></pre>
                            ${{exp.code_path ? `<a class="code-link" href="file://${{exp.code_path}}" target="_blank">📁 ${{exp.code_path}}</a>` : ''}}
                        </div>
                    </td>
                </tr>
            `).join('');

            document.getElementById('table-view').innerHTML = `
                <table>
                    <thead>
                        <tr>
                            <th>ID</th>
                            <th>Parent</th>
                            <th>Idea</th>
                            <th>AUC</th>
                            <th>TPR@1%</th>
                            <th>TPR@5%</th>
                            <th></th>
                        </tr>
                    </thead>
                    <tbody>${{rows}}</tbody>
                </table>
            `;
        }}

        function toggleTableRow(id) {{
            const row = document.getElementById(`table-details-${{id}}`);
            const wasHidden = row.classList.contains('hidden');
            row.classList.toggle('hidden');
            if (wasHidden) {{
                highlightCode();
            }}
        }}

        function updateStats() {{
            const total = experiments.length;
            const roots = getRoots(experiments).length;
            const bestAuc = Math.max(...experiments.map(e => e.auc_score || 0));
            const bestTpr1 = Math.max(...experiments.map(e => e.tpr_1_score || 0));
            const bestTpr5 = Math.max(...experiments.map(e => e.tpr_5_score || 0));

            document.getElementById('total-count').textContent = total;
            document.getElementById('root-count').textContent = roots;
            document.getElementById('best-auc').textContent = bestAuc.toFixed(4);
            document.getElementById('best-tpr1').textContent = bestTpr1.toFixed(4);
            document.getElementById('best-tpr5').textContent = bestTpr5.toFixed(4);
        }}

        function highlightCode() {{
            document.querySelectorAll('pre code:not(.hljs)').forEach((block) => {{
                hljs.highlightElement(block);
            }});
        }}

        function render() {{
            const isTreeView = document.getElementById('tree-btn').classList.contains('active');
            if (isTreeView) {{
                renderTreeView();
            }} else {{
                renderTableView();
            }}
            highlightCode();
        }}

        // Event listeners
        document.getElementById('search').addEventListener('input', render);
        document.getElementById('sort-by').addEventListener('change', render);
        document.getElementById('sort-order').addEventListener('change', render);
        document.getElementById('min-auc').addEventListener('input', render);

        document.getElementById('tree-btn').addEventListener('click', () => {{
            document.getElementById('tree-btn').classList.add('active');
            document.getElementById('table-btn').classList.remove('active');
            document.getElementById('tree-view').classList.remove('hidden');
            document.getElementById('table-view').classList.add('hidden');
            render();
        }});

        document.getElementById('table-btn').addEventListener('click', () => {{
            document.getElementById('table-btn').classList.add('active');
            document.getElementById('tree-btn').classList.remove('active');
            document.getElementById('table-view').classList.remove('hidden');
            document.getElementById('tree-view').classList.add('hidden');
            render();
        }});

        // Initial render
        updateStats();
        render();
    </script>
</body>
</html>
'''
    return html_content


def main():
    args = parse_args()

    print(f"Fetching experiments from table '{args.table_name}' (backend={args.backend})...")
    if args.backend == "file":
        if args.data_dir is None:
            args.data_dir = Path(args.output_dir) / "database"
        experiments = fetch_experiments_from_files(args.data_dir, args.table_name)
    else:
        experiments = fetch_experiments(args.db_name, args.db_user, args.db_password, args.table_name)
    print(f"Found {len(experiments)} experiments")

    print(f"Scanning runtime directory '{args.output_dir}/runtime'...")
    runtime_files = find_runtime_files(args.output_dir)
    print(f"Found {len(runtime_files)} runtime files")

    print("Generating HTML dashboard...")
    html_content = generate_html(experiments, runtime_files, args.output_dir)

    output_path = Path(args.output_dir) / "dashboard.html"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html_content)
    print(f"Dashboard saved to: {output_path.absolute()}")


if __name__ == "__main__":
    main()
