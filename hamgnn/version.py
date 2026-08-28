
"""
Version information for HamGNN.

This module provides version information for the HamGNN package, including
optional Git commit and working-tree status information.
"""
import os
import subprocess
from datetime import datetime

# Hard-coded version - updated with each release
__version__ = "2.2.0"

# Descriptive version string with build info
VERSION = __version__

# Logo with version info
soft_logo = f"""
##################################################################
#                                                                #
#    ██╗  ██╗ █████╗ ███╗   ███╗ ██████╗ ███╗   ██╗███╗   ██╗    #
#    ██║  ██║██╔══██╗████╗ ████║██╔════╝ ████╗  ██║████╗  ██║    #
#    ███████║███████║██╔████╔██║██║  ███╗██╔██╗ ██║██╔██╗ ██║    #
#    ██╔══██║██╔══██║██║╚██╔╝██║██║   ██║██║╚██╗██║██║╚██╗██║    #
#    ██║  ██║██║  ██║██║ ╚═╝ ██║╚██████╔╝██║ ╚████║██║ ╚████║    #
#    ╚═╝  ╚═╝╚═╝  ╚═╝╚═╝     ╚═╝ ╚═════╝ ╚═╝  ╚═══╝╚═╝  ╚═══╝    #
#                       Version: {VERSION}                           #
#        Author: Yang Zhong     Email: yzhong@fudan.edu.cn       #
##################################################################
"""

def get_git_version():
    """
    Get commit and working-tree status from Git if available.
    
    Returns
    -------
    tuple
        (commit, is_dirty_flag)
    """
    try:
        # Get the path to the directory containing this file
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        
        # Check if it's a git repository
        if not os.path.exists(os.path.join(base_dir, '.git')):
            return None, False
        
        # Get current commit hash
        try:
            commit = subprocess.check_output(
                ['git', 'rev-parse', '--short', 'HEAD'],
                cwd=base_dir,
                stderr=subprocess.STDOUT
            ).decode('utf-8').strip()
        except (subprocess.CalledProcessError, OSError, UnicodeDecodeError):
            commit = None

        # Check if the working directory is dirty independently of commit lookup.
        try:
            status = subprocess.check_output(
                ['git', 'status', '--porcelain'],
                cwd=base_dir,
                stderr=subprocess.STDOUT
            ).decode('utf-8').strip()
            is_dirty = bool(status)
        except (subprocess.CalledProcessError, OSError, UnicodeDecodeError):
            is_dirty = False

        if commit is None and not is_dirty:
            return None, False
        return commit, is_dirty
    except (subprocess.CalledProcessError, OSError, UnicodeDecodeError):
        return None, False

def get_version_info():
    """
    Get detailed version information.
    
    Returns
    -------
    dict
        Dictionary containing version information
    """
    commit, is_dirty = get_git_version()

    return {
        'version': __version__,
        # Keep git_version as a compatibility alias for existing consumers.
        'git_version': commit,
        'commit': commit,
        'release': True,
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'is_dirty': is_dirty
    }

# Runtime and package versions always come from the explicit release value.
version_info = get_version_info()
VERSION = __version__

# Update logo with the full version
soft_logo = soft_logo.replace(f"Version: {__version__}", f"Version: {VERSION}")

def get_version():
    """Return the version string."""
    return VERSION

def get_full_version_info():
    """Return the complete version information dictionary."""
    return version_info
