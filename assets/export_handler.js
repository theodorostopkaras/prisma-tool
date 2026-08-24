/**
 * KOSMA-τ Grid Explorer - Export Format Handler
 * 
 * Updates all Plotly graphs to use the selected export format (PNG or SVG)
 * when clicking the camera icon in the modebar.
 */

(function() {
    'use strict';
    
    let currentFormat = 'png';
    
    // Update the export config on all Plotly graphs
    function updateAllGraphs(format) {
        currentFormat = format;
        const scale = (format === 'svg') ? 1 : 2;
        
        document.querySelectorAll('.js-plotly-plot').forEach(function(gd) {
            try {
                // Update the context config that the modebar button reads
                if (gd._context) {
                    gd._context.toImageButtonOptions = {
                        format: format,
                        scale: scale,
                        filename: (gd.id || 'kosma_plot').replace(/[^a-zA-Z0-9_-]/g, '_')
                    };
                }
                
                // Also try updating via fullLayout
                if (gd._fullLayout && gd._fullLayout._config) {
                    gd._fullLayout._config.toImageButtonOptions = {
                        format: format,
                        scale: scale,
                        filename: (gd.id || 'kosma_plot').replace(/[^a-zA-Z0-9_-]/g, '_')
                    };
                }
            } catch (e) {
                // Silently ignore errors for graphs that aren't fully initialized
            }
        });
        
        console.log('[KOSMA Export] All graphs updated to:', format);
    }
    
    // Listen for radio button changes
    function setupListener() {
        document.body.addEventListener('change', function(e) {
            if (e.target.type === 'radio') {
                const selector = e.target.closest('#export-format-selector');
                if (selector && e.target.checked) {
                    updateAllGraphs(e.target.value);
                }
            }
        });
        
        // Also listen for click events as backup (some Dash components use synthetic events)
        document.body.addEventListener('click', function(e) {
            const label = e.target.closest('#export-format-selector label');
            if (label) {
                const input = label.querySelector('input[type="radio"]');
                if (input) {
                    setTimeout(function() {
                        if (input.checked) {
                            updateAllGraphs(input.value);
                        }
                    }, 50);
                }
            }
        });
    }
    
    // Initial setup and periodic refresh
    function init() {
        console.log('[KOSMA Export] Initializing export format handler...');
        
        setupListener();
        
        // Check initial state
        function checkInitial() {
            const selector = document.getElementById('export-format-selector');
            if (selector) {
                const checked = selector.querySelector('input[type="radio"]:checked');
                if (checked) {
                    updateAllGraphs(checked.value);
                }
            }
        }
        
        // Run periodically to catch new graphs and ensure config is applied
        setInterval(function() {
            checkInitial();
        }, 2000);
        
        // Initial check after a short delay
        setTimeout(checkInitial, 500);
        setTimeout(checkInitial, 1500);
        
        console.log('[KOSMA Export] Handler ready');
    }
    
    // Start when DOM is ready
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', init);
    } else {
        init();
    }
})();
