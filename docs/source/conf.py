# docs/conf.py
import os
import sys
from pathlib import Path
import yaml
# -- Path setup --------------------------------------------------------------
# Add the project root directory to sys.path so Sphinx can find the HamGNN_v_2_1 module
sys.path.insert(0, os.path.abspath('../..'))
# -- Project information -----------------------------------------------------
project = 'HamGNN_v_2_1'
copyright = 'HamGNN Team'
author = 'HamGNN Team'
release = '2.1'
# -- General configuration ---------------------------------------------------
extensions = [
    'sphinx.ext.autodoc',          # Automatically generate documentation from docstrings
    'sphinx.ext.napoleon',         # Parse Google-style and NumPy-style docstrings
    'sphinx.ext.viewcode',         # Add source code links in documentation
    'sphinx_autodoc_typehints',    # Render type hints in documentation
    'myst_parser',                 # Support Markdown files (.md)
    'sphinx.ext.intersphinx',      # Link to documentation of other projects (Python, NumPy, PyTorch)
    'sphinx.ext.mathjax',          # Render LaTeX equations
    'sphinx_copybutton',           # Code copy button
    'sphinx.ext.graphviz',         # Generate class inheritance diagrams
]
# MyST parser configuration
myst_enable_extensions = [
    "html_admonition",
    "dollarmath",  # Support LaTeX math formulas $...$ and $$...$$
]
myst_heading_anchors = 3
# Autodoc configuration - optimized settings
autodoc_member_order = "bysource"
autosummary_generate = True
source_suffix = [".rst", ".md"]
templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store', '**/toolbox/**', 'HamGNN_v_2_1/toolbox/**', 'TRANSLATION_README.md']
# Additional exclusion rules for multi-version builds
build_all_docs = os.environ.get("build_all_docs")
if build_all_docs:
    current_version = os.environ.get("current_version", "v2.1")
    current_source_dir = os.environ.get("current_source_dir", "")
    
    # Use main index file
    master_doc = 'index'
    
    # Exclude v1.0 content when building v2.1
    if current_version == "v2.1":
        exclude_patterns.extend(['source_v1/**', 'source_v1'])
    # Exclude v2.0 content when building v1.0
    elif current_version == "v1.0":
        exclude_patterns.extend(['source_v2/**', 'source_v2'])
else:
    # Use main index for single version build
    master_doc = 'index'
language = os.environ.get('HAMGNN_DOC_LANGUAGE', 'en')
# Multilingual support configuration
locale_dirs = ['../locale/']   # Translation files directory
gettext_compact = False     # Generate separate .pot files for each document
gettext_uuid = True         # Use UUID to track translations
gettext_location = False    # Don't include line numbers in .pot files
# -- Options for HTML output -------------------------------------------------
# Use RTD theme, which natively supports version switcher
html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']
html_title = "HamGNN v2.1 Documentation"
html_logo = "_static/logo.png"
html_favicon = "_static/logo.png"
html_css_files = ["custom.css"]
html_copy_source = True
html_show_sourcelink = True
html_sourcelink_suffix = ''
# GitHub source code link configuration
html_context = {
    "display_github": True,
    "github_user": "bud-primordium",
    "github_repo": "HamGNN_2_1_temp",
    "github_version": "main",
    "conf_py_path": "/docs/",
}
# RTD theme configuration - fully utilize its features
html_theme_options = {
    # Navigation bar configuration
    'navigation_depth': 4,  # Show deeper levels of navigation
    'collapse_navigation': False,  # Expand navigation by default
    'sticky_navigation': True,  # Fix navigation bar when scrolling
    'includehidden': True,
    'titles_only': False,  # Show subtitles
    
    # Display configuration
    'prev_next_buttons_location': 'both',  # Show previous/next buttons at top and bottom
    
    # Style configuration
    'style_nav_header_background': '#2980B9',  # Navigation bar background color
    'style_external_links': True,  # Add icon for external links
    
    # Logo configuration
    'logo_only': False,
}
# Code copy button configuration
copybutton_prompt_text = r">>> |\.\.\. |\$ |In \[\d*\]: | {2,5}\.\.\.: | {5,8}: "
copybutton_prompt_is_regexp = True
copybutton_exclude = '.linenos, .gp'
# -- Intersphinx configuration -----------------------------------------------
# Configure cross-project documentation links
intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
    'numpy': ('https://numpy.org/doc/stable/', None),
    'torch': ('https://pytorch.org/docs/stable/', None),
    'pytorch_lightning': ('https://lightning.ai/docs/pytorch/stable/', None),
    'e3nn': ('https://docs.e3nn.org/en/stable/', None),
    'torchmetrics': ('https://lightning.ai/docs/torchmetrics/stable/', None),
    'ase': ('https://wiki.fysik.dtu.dk/ase/', None),
}
# -- Multi-version configuration ----------------------------------------------
# Version and language configuration (for custom multi-version system)
def _normalize_language_code(code: str) -> str:
    if not code:
        return "en"
    code = str(code).lower()
    if code.startswith("zh"):
        return "zh"
    if code.startswith("en"):
        return "en"
    return code

def _build_language_base(
    branch: str,
    version: str,
    code_raw: str | None,
    source_dir: str,
    pages_root: str,
    build_all_docs_mode: bool,
) -> str:
    code_normalized = _normalize_language_code(code_raw)
    code_segment = code_raw or code_normalized or "en"
    def _join(parts):
        cleaned = []
        for part in parts:
            if not part:
                continue
            cleaned.append(str(part).strip("/"))
        return "/".join(cleaned)
    # Always use complete directory structure: /branch/version/language/ (excluding source_dir)
    return "/" + _join([
        pages_root,
        branch,
        version,
        code_segment,
    ]) + "/"

def _build_language_url(
    branch: str,
    version: str,
    language_code: str,
    source_dir: str,
    pages_root: str,
    build_all_docs_flag: bool,
) -> str:
    """Build URL for language switching"""
    return _build_language_base(
        branch=branch,
        version=version,
        code_raw=language_code,
        source_dir=source_dir,
        pages_root=pages_root,
        build_all_docs_mode=build_all_docs_flag,
    )

VERSIONS_CONFIG_PATH = Path(__file__).resolve().parent.parent / "versions.yaml"
try:
    with VERSIONS_CONFIG_PATH.open("r", encoding="utf-8") as versions_file:
        _versions_config_raw = yaml.safe_load(versions_file) or {}
except (FileNotFoundError, yaml.YAMLError):
    _versions_config_raw = {}
_branches_config = _versions_config_raw.get("branches", {})
CURRENT_VERSION = os.environ.get('HAMGNN_DOC_VERSION', 'v2.1')
CURRENT_LANGUAGE_RAW = os.environ.get('HAMGNN_DOC_LANGUAGE') or (language or 'en')
CURRENT_BRANCH = os.environ.get('SPHINX_MULTIVERSION_NAME') or os.environ.get('GITHUB_REF_NAME') or 'docs'

def _resolve_context(branch: str, version: str, language_code: str):
    normalized_language = _normalize_language_code(language_code)
    branch_conf = _branches_config.get(branch, {})
    branch_display = branch_conf.get("display_name", branch)
    versions_conf = branch_conf.get("versions", {})
    version_conf = versions_conf.get(version, {})
    version_display = version_conf.get("display_name", version)
    languages_conf = version_conf.get("languages", [])
    language_display = normalized_language
    for entry in languages_conf:
        entry_code = _normalize_language_code(entry.get("code"))
        if entry_code == normalized_language:
            language_display = entry.get("name", language_display)
            break
    return {
        "branch": branch,
        "branch_display": branch_display,
        "versions_conf": versions_conf,
        "version": version,
        "version_display": version_display,
        "languages_conf": languages_conf,
        "language": normalized_language,
        "language_display": language_display,
    }

_current_context = _resolve_context(CURRENT_BRANCH, CURRENT_VERSION, CURRENT_LANGUAGE_RAW)
html_context.setdefault('branches', [])
html_context.setdefault('versions', [])
html_context.setdefault('languages', [])
html_context.update({
    'current_branch': _current_context['branch'],
    'current_branch_display': _current_context['branch_display'],
    'current_version': _current_context['version'],
    'current_version_display': _current_context['version_display'],
    'current_language': _current_context['language'],
    'current_language_display': _current_context['language_display'],
})
_current_branch_conf = _branches_config.get(_current_context['branch'], {})
_current_version_conf = _current_branch_conf.get('versions', {}).get(
    _current_context['version'], {}
)
_current_source_dir = _current_version_conf.get('source_dir', 'source')
build_all_docs = os.environ.get("build_all_docs")
pages_root = os.environ.get("pages_root", "")
build_all_docs_flag = bool(build_all_docs)
_current_output_dir = _current_source_dir if build_all_docs_flag else ""
html_context['build_all_docs_mode'] = build_all_docs_flag
html_context['pages_root'] = pages_root or ""
branch_list_default = [[
    _current_context['branch'],
    _current_context['branch_display'],
    'index.html',
]]
version_list_default = [[
    _current_context['version'],
    _current_context['version_display'],
    'index.html',
]]
language_entries = _current_version_conf.get('languages', [])
if language_entries:
    language_list_default = [
        {
            'code': entry.get('code'),
            'name': entry.get('name', entry.get('code')),
            'branch': _current_context['branch'],
            'version': _current_context['version'],
            'source_dir': _current_output_dir,
            'is_current': _normalize_language_code(entry.get('code')) == _current_context['language'],
            'url': _build_language_url(
                _current_context['branch'],
                _current_context['version'],
                entry.get('code'),
                _current_output_dir,
                pages_root or "",
                build_all_docs_flag,
            ),
        }
        for entry in language_entries
        if entry.get('code')
    ]
else:
    language_list_default = [
        {
            'code': _current_context['language'],
            'name': _current_context['language_display'],
            'branch': _current_context['branch'],
            'version': _current_context['version'],
            'source_dir': _current_output_dir,
            'is_current': True,
            'url': _build_language_url(
                _current_context['branch'],
                _current_context['version'],
                _current_context['language'],
                _current_output_dir,
                pages_root or "",
                build_all_docs_flag,
            ),
        }
    ]
html_context['branches'] = branch_list_default
html_context['versions'] = version_list_default
html_context['languages'] = language_list_default
if build_all_docs_flag and _versions_config_raw:
    current_branch = os.environ.get("current_branch", _current_context['branch'])
    current_version = os.environ.get("current_version", _current_context['version'])
    current_language_raw = os.environ.get("current_language", _current_context['language'])
    current_source_dir = os.environ.get("current_source_dir", _current_source_dir)
    build_context = _resolve_context(current_branch, current_version, current_language_raw)
    version = build_context['version']
    release = build_context['version']
    html_context.update({
        'current_branch': build_context['branch'],
        'current_branch_display': build_context['branch_display'],
        'current_version': build_context['version'],
        'current_version_display': build_context['version_display'],
        'current_language': build_context['language'],
        'current_language_display': build_context['language_display'],
        'build_all_docs_mode': True,
        'pages_root': pages_root or "",
    })
    branch_conf = _branches_config.get(build_context['branch'], {})
    version_conf = branch_conf.get('versions', {}).get(build_context['version'], {})
    version_source_dir = version_conf.get('source_dir', current_source_dir)
    branch_list = []
    for branch_key, branch_info in _branches_config.items():
        branch_versions = branch_info.get('versions', {})
        if build_context['version'] not in branch_versions:
            continue
        target_source_dir = branch_versions[build_context['version']].get('source_dir', version_source_dir)
        branch_url = _build_language_url(
            branch_key,
            build_context['version'],
            build_context['language'],
            target_source_dir,
            pages_root or "",
            True,
        )
        branch_list.append([
            branch_key,
            branch_info.get('display_name', branch_key),
            'index.html' if branch_key == build_context['branch'] else branch_url,
        ])
    if branch_list:
        html_context['branches'] = branch_list
    version_list = []
    for version_key, version_info in branch_conf.get('versions', {}).items():
        target_source_dir = version_info.get('source_dir', version_source_dir)
        version_url = _build_language_url(
            build_context['branch'],
            version_key,
            build_context['language'],
            target_source_dir,
            pages_root or "",
            True,
        )
        version_list.append([
            version_key,
            version_info.get('display_name', version_key),
            'index.html' if version_key == build_context['version'] else version_url,
        ])
    if version_list:
        html_context['versions'] = version_list
    lang_list = []
    for language_entry in version_conf.get('languages', []):
        lang_code_raw = language_entry.get('code')
        if not lang_code_raw:
            continue
        lang_list.append({
            'code': lang_code_raw,
            'name': language_entry.get('name', lang_code_raw),
            'branch': build_context['branch'],
            'version': build_context['version'],
            'source_dir': version_source_dir,
            'is_current': _normalize_language_code(lang_code_raw) == build_context['language'],
            'url': _build_language_url(
                build_context['branch'],
                build_context['version'],
                lang_code_raw,
                version_source_dir,
                pages_root or "",
                True,
            ),
        })
    if lang_list:
        html_context['languages'] = lang_list
# -- Custom event handler to skip specific headers -----------------------------
def remove_custom_header(app, what, name, obj, options, lines):
    """
    Called when Sphinx processes docstrings, used to remove specific file headers.
    """
    # Define a tuple containing all header "fingerprints" to be identified and removed
    header_signatures = (
        "Descripttion:",
        "/*",
        "@Author:"
    )
    
    if not lines:
        return
    # Check if the first few lines contain any of the "fingerprints"
    # We concatenate the first 5 lines for checking to handle various formats
    docstring_head = "".join(lines[:5])
    for signature in header_signatures:
        if signature in docstring_head:
            lines.clear()
            # Once matched, clear and return immediately
            return
def setup(app):
    """
    Register our custom handler to Sphinx's event manager.
    """
    app.connect('autodoc-process-docstring', remove_custom_header)