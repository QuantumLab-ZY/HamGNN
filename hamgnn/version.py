
"""
Version information for HamGNN.

This module provides version information for the HamGNN package, including
functions to extract version from Git if available.
"""
import os
import subprocess
import re
from datetime import datetime

# Hard-coded version - updated with each release
__version__ = "2.1.0"

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
    Get version from git if available.
    
    Returns
    -------
    tuple
        (version_string, is_dirty_flag)
    """
    try:
        # Get the path to the directory containing this file
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        
        # Check if it's a git repository
        if not os.path.exists(os.path.join(base_dir, '.git')):
            return None, False
        
        # Get the most recent tag
        try:
            tag = subprocess.check_output(
                ['git', 'describe', '--tags', '--abbrev=0'],
                cwd=base_dir, stderr=subprocess.STDOUT
            ).decode('utf-8').strip()
        except subprocess.CalledProcessError:
            tag = '0.0.0'
        
        # Get current commit hash
        commit = subprocess.check_output(
            ['git', 'rev-parse', '--short', 'HEAD'],
            cwd=base_dir
        ).decode('utf-8').strip()
        
        # Check if working directory is dirty
        status = subprocess.check_output(
            ['git', 'status', '--porcelain'],
            cwd=base_dir
        ).decode('utf-8').strip()
        is_dirty = len(status) > 0
        
        # Get commit count since tag
        try:
            count = subprocess.check_output(
                ['git', 'rev-list', f'{tag}..HEAD', '--count'],
                cwd=base_dir
            ).decode('utf-8').strip()
        except subprocess.CalledProcessError:
            count = '0'
        
        if count == '0':
            version = tag
        else:
            version = f"{tag}+{count}.{commit}"
        
        # Add dirty flag if needed
        if is_dirty:
            version += "-dirty"
            
        return version, is_dirty
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
    git_version, is_dirty = get_git_version()
    if git_version:
        version = git_version
    else:
        version = __version__
        is_dirty = False
    
    # Determine if this is a release version
    is_release = bool(re.match(r'^[0-9]+\.[0-9]+\.[0-9]+$', version))
    
    return {
        'version': version,
        'git_version': git_version,
        'release': is_release,
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'is_dirty': is_dirty
    }

# Update the VERSION with git info if available
version_info = get_version_info()
VERSION = version_info['version']

# Update logo with the full version
soft_logo = soft_logo.replace(f"Version: {__version__}", f"Version: {VERSION}")

def get_version():
    """Return the version string."""
    return VERSION

def get_full_version_info():
    """Return the complete version information dictionary."""
    return version_info
