const form = document.getElementById("predictionForm");
const resultCard = document.getElementById("resultCard");
const priceValue = document.getElementById("priceValue");
const agentMessage = document.getElementById("agentMessage");
const errorMessage = document.getElementById("errorMessage");
const predictButton = document.getElementById("predictButton");
const yearInput = document.getElementById("year");

const currentYear = new Date().getFullYear();
yearInput.max = String(currentYear + 1);
yearInput.value = String(currentYear - 2);

function setLoading(isLoading) {
  predictButton.disabled = isLoading;
  predictButton.textContent = isLoading ? "Analyzing..." : "Get Smart Price";
}

function showError(message) {
  resultCard.classList.remove("hidden");
  errorMessage.textContent = message;
  errorMessage.classList.remove("hidden");
  priceValue.textContent = "--";
  agentMessage.textContent = "Please adjust inputs and try again.";
}

function clearError() {
  errorMessage.textContent = "";
  errorMessage.classList.add("hidden");
}

function formatCurrency(value) {
  return new Intl.NumberFormat("en-US", {
    style: "currency",
    currency: "USD",
    maximumFractionDigits: 0,
  }).format(value);
}

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  clearError();
  setLoading(true);

  const payload = {
    year: Number.parseInt(document.getElementById("year").value, 10),
    milage: Number.parseInt(document.getElementById("milage").value, 10),
    engine_size: Number.parseFloat(document.getElementById("engine_size").value),
  };

  if (Number.isNaN(payload.year) || Number.isNaN(payload.milage) || Number.isNaN(payload.engine_size)) {
    setLoading(false);
    showError("All fields are required and must be valid numbers.");
    return;
  }

  try {
    const response = await fetch("/predict", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify(payload),
    });

    const data = await response.json();

    if (!response.ok) {
      throw new Error(data.error || "Prediction request failed.");
    }

    resultCard.classList.remove("hidden");
    priceValue.textContent = formatCurrency(data.predicted_price);
    agentMessage.textContent = data.smart_agent_message;
  } catch (error) {
    showError(error.message || "An unexpected error happened.");
  } finally {
    setLoading(false);
  }
});
