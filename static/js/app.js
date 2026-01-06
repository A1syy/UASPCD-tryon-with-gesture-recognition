/**
 * Face Filter Web App - JavaScript
 * Mengimplementasikan interaksi UI dengan backend Flask
 */

// ==================== State Management ====================
const state = {
  masks: [],
  currentMaskIdx: 0,
  params: {
    scale: 200,
    offset_x: 0,
    offset_y: -25,
    yaw: 150,
    pitch: 150,
    roll: 0,
  },
  isPaused: false,
  showInfo: true,
  handControl: false,
  isConnected: false,
  fps: 0,
  slidersHidden: false,
};

// ==================== DOM Elements ====================
const elements = {
  // Status
  statusIndicator: document.getElementById("statusIndicator"),
  statusText: document.getElementById("statusText"),
  fpsCounter: document.getElementById("fpsCounter"),

  // Preview
  videoFeed: document.getElementById("videoFeed"),
  previewLoading: document.getElementById("previewLoading"),
  gestureHints: document.getElementById("gestureHints"),

  // Quick Actions
  pauseBtn: document.getElementById("pauseBtn"),
  resetBtn: document.getElementById("resetBtn"),
  clearMaskBtn: document.getElementById("clearMaskBtn"),

  // Current Mask
  currentMaskPreview: document.getElementById("currentMaskPreview"),
  currentMaskName: document.getElementById("currentMaskName"),

  // Mask Grid
  maskGrid: document.getElementById("maskGrid"),

  // Sliders
  scaleSlider: document.getElementById("scaleSlider"),
  scaleValue: document.getElementById("scaleValue"),
  offsetXSlider: document.getElementById("offsetXSlider"),
  offsetXValue: document.getElementById("offsetXValue"),
  offsetYSlider: document.getElementById("offsetYSlider"),
  offsetYValue: document.getElementById("offsetYValue"),
  yawSlider: document.getElementById("yawSlider"),
  yawValue: document.getElementById("yawValue"),
  pitchSlider: document.getElementById("pitchSlider"),
  pitchValue: document.getElementById("pitchValue"),
  rollSlider: document.getElementById("rollSlider"),
  rollValue: document.getElementById("rollValue"),
  resetParamsBtn: document.getElementById("resetParamsBtn"),

  // Slider visibility
  toggleSlidersBtn: document.getElementById("toggleSlidersBtn"),
  slidersContainer: document.getElementById("slidersContainer"),

  // Toggles
  showInfoToggle: document.getElementById("showInfoToggle"),
  handControlToggle: document.getElementById("handControlToggle"),

  // Modal
  helpBtn: document.getElementById("helpBtn"),
  helpModal: document.getElementById("helpModal"),
  closeHelpModal: document.getElementById("closeHelpModal"),

  // Toast
  toast: document.getElementById("toast"),
  toastIcon: document.getElementById("toastIcon"),
  toastMessage: document.getElementById("toastMessage"),
};

// ==================== Slider Visibility ====================
function setSlidersHidden(hidden) {
  state.slidersHidden = hidden;

  if (elements.slidersContainer) {
    elements.slidersContainer.classList.toggle("hidden", hidden);
  }

  if (elements.toggleSlidersBtn) {
    elements.toggleSlidersBtn.textContent = hidden ? "👁 SHOW" : "👁 HIDE";
  }
}

// ==================== API Functions ====================
async function fetchAPI(endpoint, method = "GET", body = null) {
  try {
    const options = {
      method,
      headers: { "Content-Type": "application/json" },
    };
    if (body) {
      options.body = JSON.stringify(body);
    }
    const response = await fetch(endpoint, options);
    return await response.json();
  } catch (error) {
    console.error("API Error:", error);
    showToast("Koneksi terputus", "error");
    return { success: false, error: error.message };
  }
}

async function loadMasks() {
  const result = await fetchAPI("/api/masks");
  if (result.success) {
    state.masks = result.masks;
    state.currentMaskIdx = result.current;
    renderMaskGrid();
    updateCurrentMaskDisplay();
  }
}

async function setMask(index) {
  const result = await fetchAPI("/api/mask", "POST", { index });
  if (result.success) {
    state.currentMaskIdx = result.current;
    updateCurrentMaskDisplay();
    highlightSelectedMask();
    showToast(`Mask: ${result.mask_name}`, "success");
  }
}

async function updateParams(params) {
  const result = await fetchAPI("/api/params", "POST", params);
  return result.success;
}

async function resetParams() {
  const result = await fetchAPI("/api/reset", "POST");
  if (result.success) {
    state.params = result.params;
    updateSliderValues();
    showToast("Parameter direset", "success");
  }
}

async function toggleSetting(setting) {
  const result = await fetchAPI("/api/toggle", "POST", { setting });
  if (result.success) {
    return result.value;
  }
  return null;
}

async function getStatus() {
  const result = await fetchAPI("/api/status");
  if (result.success) {
    updateStatusDisplay(result.status);
  }
}

// ==================== UI Update Functions ====================
function updateStatusDisplay(status) {
  state.isConnected = status.running;
  state.isPaused = status.paused;
  state.showInfo = status.show_info;
  state.handControl = status.hand_control;
  state.fps = status.fps;
  state.currentMaskIdx = status.current_mask_idx;

  // Update status indicator
  if (status.running) {
    elements.statusIndicator.className =
      "status-indicator " + (status.paused ? "paused" : "live");
    elements.statusText.textContent = status.paused ? "Paused" : "Live";
  } else {
    elements.statusIndicator.className = "status-indicator error";
    elements.statusText.textContent = "Disconnected";
  }

  // Update FPS
  elements.fpsCounter.textContent = `${status.fps} FPS`;

  // Update pause button
  elements.pauseBtn.classList.toggle("active", status.paused);
  elements.pauseBtn.querySelector(".btn-icon").textContent = status.paused
    ? "▶"
    : "⏸";
  elements.pauseBtn.querySelector(".btn-text").textContent = status.paused
    ? "Resume"
    : "Pause";

  // Update toggles
  elements.showInfoToggle.checked = status.show_info;
  elements.handControlToggle.checked = status.hand_control;

  // Update gesture hints visibility
  elements.gestureHints.classList.toggle("visible", status.hand_control);

  // Update current mask name
  elements.currentMaskName.textContent = status.current_mask;
}

function renderMaskGrid() {
  elements.maskGrid.innerHTML = "";

  state.masks.forEach((mask, index) => {
    const item = document.createElement("div");
    item.className =
      "mask-item" + (index === state.currentMaskIdx ? " selected" : "");
    item.dataset.index = index;
    item.title = mask.display_name;

    if (index === 0) {
      item.innerHTML = '<span class="no-mask">🚫</span>';
    } else if (mask.thumbnail) {
      item.innerHTML = `<img src="data:image/png;base64,${mask.thumbnail}" alt="${mask.display_name}">`;
    } else {
      item.innerHTML = '<span class="no-mask">🎭</span>';
    }

    // Add name label
    const nameLabel = document.createElement("div");
    nameLabel.className = "mask-item-name";
    nameLabel.textContent = mask.display_name;
    item.appendChild(nameLabel);

    item.addEventListener("click", () => setMask(index));
    elements.maskGrid.appendChild(item);
  });
}

function highlightSelectedMask() {
  document.querySelectorAll(".mask-item").forEach((item, index) => {
    item.classList.toggle("selected", index === state.currentMaskIdx);
  });
}

function updateCurrentMaskDisplay() {
  const mask = state.masks[state.currentMaskIdx];
  if (mask) {
    elements.currentMaskName.textContent = mask.display_name;

    if (state.currentMaskIdx === 0) {
      elements.currentMaskPreview.innerHTML =
        '<span class="no-mask-icon">🎭</span>';
    } else if (mask.thumbnail) {
      elements.currentMaskPreview.innerHTML = `<img src="data:image/png;base64,${mask.thumbnail}" alt="${mask.display_name}">`;
    } else {
      elements.currentMaskPreview.innerHTML =
        '<span class="no-mask-icon">🎭</span>';
    }
  }
}

function updateSliderValues() {
  elements.scaleSlider.value = state.params.scale;
  elements.scaleValue.textContent = `${state.params.scale}%`;

  elements.offsetXSlider.value = state.params.offset_x;
  elements.offsetXValue.textContent =
    state.params.offset_x > 0
      ? `+${state.params.offset_x}`
      : state.params.offset_x;

  elements.offsetYSlider.value = state.params.offset_y;
  elements.offsetYValue.textContent =
    state.params.offset_y > 0
      ? `+${state.params.offset_y}`
      : state.params.offset_y;

  elements.yawSlider.value = state.params.yaw;
  elements.yawValue.textContent = `${state.params.yaw}%`;

  elements.pitchSlider.value = state.params.pitch;
  elements.pitchValue.textContent = `${state.params.pitch}%`;

  elements.rollSlider.value = state.params.roll;
  elements.rollValue.textContent = `${state.params.roll}°`;
}

// ==================== Toast Notification ====================
function showToast(message, type = "success") {
  elements.toastMessage.textContent = message;
  elements.toast.className = `toast ${type} visible`;

  // Set icon based on type
  const icons = {
    success: "✓",
    error: "✕",
    warning: "⚠",
  };
  elements.toastIcon.textContent = icons[type] || icons.success;

  // Hide after 2 seconds
  setTimeout(() => {
    elements.toast.classList.remove("visible");
  }, 2000);
}

// ==================== Event Handlers ====================
function setupEventListeners() {
  // Video feed load handler
  elements.videoFeed.addEventListener("load", () => {
    elements.previewLoading.classList.add("hidden");
  });

  elements.videoFeed.addEventListener("error", () => {
    elements.previewLoading.querySelector("p").textContent =
      "Gagal memuat kamera";
  });

  // Quick action buttons
  elements.pauseBtn.addEventListener("click", async () => {
    const value = await toggleSetting("pause");
    if (value !== null) {
      state.isPaused = value;
      elements.pauseBtn.classList.toggle("active", value);
      elements.pauseBtn.querySelector(".btn-icon").textContent = value
        ? "▶"
        : "⏸";
      elements.pauseBtn.querySelector(".btn-text").textContent = value
        ? "Resume"
        : "Pause";
      showToast(value ? "Video dijeda" : "Video dilanjutkan", "success");
    }
  });

  elements.resetBtn.addEventListener("click", () => {
    resetParams();
  });

  elements.clearMaskBtn.addEventListener("click", () => {
    setMask(0);
  });

  // Reset params button
  elements.resetParamsBtn.addEventListener("click", () => {
    resetParams();
  });

  // Hide/Show all sliders
  if (elements.toggleSlidersBtn) {
    elements.toggleSlidersBtn.addEventListener("click", () => {
      setSlidersHidden(!state.slidersHidden);
    });
  }

  // Slider event handlers with debounce
  let sliderTimeout;
  const handleSliderChange = (
    slider,
    paramKey,
    displayEl,
    suffix = "",
    prefix = ""
  ) => {
    slider.addEventListener("input", (e) => {
      const value = parseInt(e.target.value);
      state.params[paramKey] = value;

      // Update display
      let displayValue = value;
      if (prefix && value > 0) displayValue = `+${value}`;
      displayEl.textContent = `${
        prefix ? (value > 0 ? "+" : "") : ""
      }${value}${suffix}`;

      // Debounce API call
      clearTimeout(sliderTimeout);
      sliderTimeout = setTimeout(() => {
        updateParams({ [paramKey]: value });
      }, 100);
    });
  };

  handleSliderChange(elements.scaleSlider, "scale", elements.scaleValue, "%");
  handleSliderChange(
    elements.offsetXSlider,
    "offset_x",
    elements.offsetXValue,
    "",
    "+"
  );
  handleSliderChange(
    elements.offsetYSlider,
    "offset_y",
    elements.offsetYValue,
    "",
    "+"
  );
  handleSliderChange(elements.yawSlider, "yaw", elements.yawValue, "%");
  handleSliderChange(elements.pitchSlider, "pitch", elements.pitchValue, "%");
  handleSliderChange(elements.rollSlider, "roll", elements.rollValue, "°");

  // Toggle switches
  elements.showInfoToggle.addEventListener("change", async () => {
    const value = await toggleSetting("info");
    if (value !== null) {
      state.showInfo = value;
      showToast(
        value ? "Info overlay aktif" : "Info overlay nonaktif",
        "success"
      );
    }
  });

  elements.handControlToggle.addEventListener("change", async () => {
    const value = await toggleSetting("hand_control");
    if (value !== null) {
      state.handControl = value;
      elements.gestureHints.classList.toggle("visible", value);
      showToast(
        value ? "Kontrol gesture aktif" : "Kontrol gesture nonaktif",
        "success"
      );
    }
  });

  // Help modal
  elements.helpBtn.addEventListener("click", () => {
    elements.helpModal.classList.add("visible");
  });

  elements.closeHelpModal.addEventListener("click", () => {
    elements.helpModal.classList.remove("visible");
  });

  elements.helpModal.addEventListener("click", (e) => {
    if (e.target === elements.helpModal) {
      elements.helpModal.classList.remove("visible");
    }
  });

  // Keyboard shortcuts
  document.addEventListener("keydown", (e) => {
    // Don't trigger if typing in an input
    if (e.target.tagName === "INPUT") return;

    switch (e.key.toLowerCase()) {
      case " ": // Spacebar - Pause/Resume
        e.preventDefault();
        elements.pauseBtn.click();
        break;
      case "r": // Reset params
        elements.resetBtn.click();
        break;
      case "c": // Clear mask
        elements.clearMaskBtn.click();
        break;
      case "arrowleft": // Previous mask
        e.preventDefault();
        if (state.currentMaskIdx > 0) {
          setMask(state.currentMaskIdx - 1);
        } else {
          setMask(state.masks.length - 1);
        }
        break;
      case "arrowright": // Next mask
        e.preventDefault();
        if (state.currentMaskIdx < state.masks.length - 1) {
          setMask(state.currentMaskIdx + 1);
        } else {
          setMask(0);
        }
        break;
      case "g": // Toggle gesture control
        elements.handControlToggle.click();
        break;
      case "i": // Toggle info overlay
        elements.showInfoToggle.click();
        break;
      case "h": // Help modal
        if (elements.helpModal.classList.contains("visible")) {
          elements.helpModal.classList.remove("visible");
        } else {
          elements.helpModal.classList.add("visible");
        }
        break;
      case "escape": // Close modal
        elements.helpModal.classList.remove("visible");
        break;
    }
  });
}

// ==================== Status Polling ====================
function startStatusPolling() {
  // Poll status every 1 second
  setInterval(async () => {
    await getStatus();
  }, 1000);
}

// ==================== Initialize ====================
async function init() {
  console.log("🎭 Face Filter Web App Initializing...");

  // Setup event listeners
  setupEventListeners();

  // Load initial data
  await loadMasks();
  await getStatus();

  // Update slider displays
  updateSliderValues();

  // Ensure sliders start visible
  setSlidersHidden(false);

  // Start status polling
  startStatusPolling();

  // Hide loading after video starts
  setTimeout(() => {
    if (
      elements.previewLoading.querySelector("p").textContent ===
      "Memuat kamera..."
    ) {
      elements.previewLoading.classList.add("hidden");
    }
  }, 3000);

  console.log("✅ Face Filter Web App Ready");
}

// Start the app when DOM is ready
document.addEventListener("DOMContentLoaded", init);
