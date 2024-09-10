// static/js/modal.js

function openModal() {
  document.getElementById("uploadModal").classList.remove("hidden");
}

function closeModal() {
  document.getElementById("uploadModal").classList.add("hidden");
}

function uploadFile() {
  const fileInput = document.getElementById("file-input");
  const formData = new FormData();
  formData.append("file", fileInput.files[0]);

  const stepsList = document.getElementById("steps-list");
  const cleaningSteps = [
    "Step 1: Checking for missing values...",
    "Step 2: Removing duplicates...",
    "Step 3: Formatting date columns...",
    "Step 4: Standardizing text fields...",
    "Step 5: Validating data consistency...",
  ];

  document.getElementById("cleaning-steps").classList.remove("hidden");

  // Simulate the cleaning steps with delays
  cleaningSteps.forEach((step, index) => {
    setTimeout(() => {
      const li = document.createElement("li");
      li.textContent = step;
      stepsList.appendChild(li);
    }, index * 1000); // Adjust delay as needed
  });

  // Perform the actual upload after the last step
  setTimeout(() => {
    fetch("/upload", {
      method: "POST",
      body: formData,
    })
      .then((response) => response.json())
      .then((data) => {
        if (data.success) {
          window.location.href = `/`;
        } else {
          alert("File upload failed. Please try again.");
        }
      })
      .catch((error) => console.error("Error:", error));
  }, cleaningSteps.length * 1000);
}
