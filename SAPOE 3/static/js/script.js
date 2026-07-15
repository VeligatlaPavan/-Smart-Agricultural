document.addEventListener("DOMContentLoaded", () => {
    // 1. Presets Handler
    const presetButtons = document.querySelectorAll(".preset-btn");
    
    // Check if preset buttons and form elements exist before assigning events
    if (presetButtons.length > 0) {
        presetButtons.forEach(button => {
            button.addEventListener("click", () => {
                const n = button.getAttribute("data-n");
                const p = button.getAttribute("data-p");
                const k = button.getAttribute("data-k");
                const temp = button.getAttribute("data-temp");
                const humidity = button.getAttribute("data-humidity");
                const ph = button.getAttribute("data-ph");
                const rainfall = button.getAttribute("data-rainfall");
                
                // Get form inputs
                const nInput = document.getElementById("N");
                const pInput = document.getElementById("P");
                const kInput = document.getElementById("K");
                const tempInput = document.getElementById("temp");
                const humidityInput = document.getElementById("humidity");
                const phInput = document.getElementById("pH");
                const rainfallInput = document.getElementById("rainfall");
                
                // Populate fields
                if (nInput) nInput.value = n;
                if (pInput) pInput.value = p;
                if (kInput) kInput.value = k;
                if (tempInput) tempInput.value = temp;
                if (humidityInput) humidityInput.value = humidity;
                if (phInput) phInput.value = ph;
                if (rainfallInput) rainfallInput.value = rainfall;
                
                // Visual feedback: briefly flash the inputs
                const inputs = [nInput, pInput, kInput, tempInput, humidityInput, phInput, rainfallInput];
                inputs.forEach(input => {
                    if (input) {
                        input.style.borderColor = "var(--color-primary-light)";
                        input.style.backgroundColor = "var(--color-bg-mint)";
                        setTimeout(() => {
                            input.style.borderColor = "";
                            input.style.backgroundColor = "";
                        }, 500);
                    }
                });
            });
        });
    }

    // 2. Loading Overlay Handler
    const recommendForm = document.getElementById("recommendForm");
    const loadingOverlay = document.getElementById("loadingOverlay");
    
    if (recommendForm && loadingOverlay) {
        recommendForm.addEventListener("submit", (e) => {
            // HTML5 built-in validation takes care of basic validity check
            if (!recommendForm.checkValidity()) {
                return; // Let browser show standard validation tooltips
            }
            
            // Show custom loading overlay
            loadingOverlay.style.display = "flex";
            
            // Allow the form to submit naturally
        });
    }

    // 3. Result Page Gauge Animation
    const gaugeFill = document.querySelector(".gauge-fill");
    if (gaugeFill) {
        const percent = parseFloat(gaugeFill.getAttribute("data-percent"));
        // Standard circle circumference: 2 * PI * r = 2 * 3.14159 * 80 = 502.65
        const circumference = 502.65;
        // Calculate stroke offset
        const offset = circumference - (percent / 100) * circumference;
        
        // Apply with a minor delay to trigger CSS transition
        setTimeout(() => {
            gaugeFill.style.strokeDashoffset = offset;
        }, 150);
    }
});
