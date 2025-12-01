document.addEventListener('DOMContentLoaded', () => {
    const inputsContainer = document.getElementById('inputsContainer');
    const form = document.getElementById('predictionForm');
    const resultDiv = document.getElementById('result');
    const predictionText = document.getElementById('predictionText');
    const probabilityBar = document.getElementById('probabilityBar');
    const probabilityValue = document.getElementById('probabilityValue');

    // Render inputs dynamically based on COLUMNS
    if (typeof COLUMNS !== 'undefined' && COLUMNS.length > 0) {
        COLUMNS.forEach(col => {
            const formGroup = document.createElement('div');
            formGroup.className = 'form-group';

            const label = document.createElement('label');
            label.htmlFor = col;
            label.textContent = col;

            const input = document.createElement('input');
            input.type = 'number';
            input.id = col;
            input.name = col;
            input.required = true;
            input.step = 'any'; // Allow decimals
            input.placeholder = `Enter ${col}`;

            // Add some basic validation/constraints for common heart columns if detected
            if (col === 'sex') {
                input.placeholder = "0 (F) or 1 (M)";
                input.min = 0;
                input.max = 1;
            } else if (col === 'age') {
                input.min = 1;
                input.max = 120;
            }

            formGroup.appendChild(label);
            formGroup.appendChild(input);
            inputsContainer.appendChild(formGroup);
        });
    } else {
        inputsContainer.innerHTML = '<p class="error">No columns definitions found. Please train the model first.</p>';
        document.getElementById('predictBtn').disabled = true;
    }

    form.addEventListener('submit', async (e) => {
        e.preventDefault();

        // Collect data
        const formData = {};
        const inputs = inputsContainer.querySelectorAll('input');

        inputs.forEach(input => {
            formData[input.name] = parseFloat(input.value);
        });

        try {
            const response = await fetch('/predict', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify(formData)
            });

            const result = await response.json();

            if (!response.ok) {
                throw new Error(result.error || 'Prediction failed');
            }

            // Display Result
            resultDiv.classList.remove('hidden');

            const isHighRisk = result.prediction === 1;
            const probPercent = (result.probability * 100).toFixed(1);

            if (isHighRisk) {
                predictionText.textContent = "High Risk Detected";
                predictionText.className = "prediction-text high-risk";
            } else {
                predictionText.textContent = "Low Risk Detected";
                predictionText.className = "prediction-text low-risk";
            }

            probabilityBar.style.width = `${probPercent}%`;
            probabilityValue.textContent = `Probability: ${probPercent}%`;

        } catch (error) {
            alert(`Error: ${error.message}`);
        }
    });
});
