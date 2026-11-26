// ---------------------------------------------------------------------------
// Control mode (hands / eyes / torso)
//
// Nothing is loaded until the user presses Start: the server then loads only
// the selected MediaPipe model in the background and reports progress through
// /state, which this popup polls while it is open.
// ---------------------------------------------------------------------------
const MODE_INFO = {
    hands: {
        label: "Hands",
        subject: "your hand",
        centering: "Hold your open hand up",
        help: "Move your open palm to steer. Tap thumb + index = click, tap twice = double click, touch and move = drag. Tap thumb + middle = right click, touch and move up/down = scroll."
    },
    eyes: {
        label: "Eyes",
        subject: "your face",
        centering: "Look at the center of the screen",
        help: "Turn your head to steer. Wink left eye = click, twice = double click, hold it shut = drag. Wink right eye = right click. Hold mouth open + nod = scroll."
    },
    torso: {
        label: "Torso",
        subject: "your shoulders",
        centering: "Sit naturally, facing the screen",
        help: "Lean left/right and sit up/sink to steer. Hold still = click, shrug = right click. Lean in / back = scroll."
    }
};
const MODE_KEY = "wgi-mode";

function loadSelectedMode() {
    try {
        const saved = localStorage.getItem(MODE_KEY);
        if (saved && MODE_INFO[saved]) return saved;
    } catch (e) {}
    return "hands";
}

let selectedMode = loadSelectedMode();
let lastState = null;
let pendingMode = null;       // Start pressed: mode we are waiting for the server to report

const btnStart = document.getElementById('btn-start');
const btnStop = document.getElementById('btn-stop');
const btnRecenter = document.getElementById('btn-recenter');
const btnCalibrate = document.getElementById('btn-calibrate');
const loadingBar = document.getElementById('loading-bar');
const modeButtons = document.querySelectorAll('.mode-btn');

function setStatus(message) {
    document.getElementById('status').innerText = message;
    // Status light: green = tracking, pulsing amber = busy, red = error, grey = idle.
    const m = message.toLowerCase();
    let state = "idle";
    if (m.startsWith("error") || m.includes("offline") || m.includes("can't see")) state = "err";
    else if (m.includes("loading") || m.includes("opening") || m.includes("starting")
             || m.includes("stopping") || m.includes("%") || m.includes("paused")) state = "busy";
    else if (m.startsWith("tracking")) state = "ok";
    const pill = document.getElementById('status-pill');
    if (pill) pill.dataset.state = state;
}

function setLoading(visible, progress = null) {
    loadingBar.classList.toggle('visible', visible);
    loadingBar.classList.toggle('progress', progress !== null);
    loadingBar.firstElementChild.style.width = progress !== null ? `${Math.round(progress * 100)}%` : "";
}

function renderMode() {
    modeButtons.forEach(btn => {
        const on = btn.dataset.mode === selectedMode;
        btn.classList.toggle('selected', on);
        btn.setAttribute('aria-checked', on ? 'true' : 'false');
    });
    document.getElementById('mode-help').innerText = MODE_INFO[selectedMode].help;
    // Hand pinches adapt to the user's hand on their own: no calibration step.
    btnCalibrate.classList.add('hidden-el');
}

function selectMode(mode) {
    if (!MODE_INFO[mode]) return;
    selectedMode = mode;
    try { localStorage.setItem(MODE_KEY, mode); } catch (e) {}
    renderMode();
}

function updateButtonStates(isTracking) {
    btnStart.disabled = isTracking || pendingMode !== null;
    btnStop.disabled = !isTracking && pendingMode === null;
    btnStart.classList.toggle('primary', !isTracking);
    btnStop.classList.toggle('primary', isTracking);
}

function startTracking(mode) {
    pendingMode = mode;
    updateButtonStates(false);
    setStatus(`Loading ${MODE_INFO[mode].label} model...`);
    setLoading(true);
    chrome.runtime.sendMessage({ action: "start_tracking", mode }, (res) => {
        if (chrome.runtime.lastError || !res || res.error) {
            pendingMode = null;
            setStatus((res && res.message) || "Error: WGI server is not running.");
            setLoading(false);
            updateButtonStates(false);
        }
        // Success: pollState() takes it from here.
    });
}

function describeState(s) {
    const info = MODE_INFO[s.mode] || MODE_INFO[selectedMode];
    if (!s.is_tracking) {
        if (s.model_status === "error" || s.camera_status === "error") {
            return { text: `Error: ${s.error || "tracking stopped"}`, loading: false };
        }
        return { text: "Stopped", loading: false };
    }
    if (s.model_status === "loading" && s.camera_status === "opening") {
        return { text: `Loading ${info.label} model & opening camera...`, loading: true };
    }
    if (s.model_status === "loading") {
        return { text: `Loading ${info.label} model...`, loading: true };
    }
    if (s.camera_status === "opening") {
        return { text: "Opening camera (can take a while)...", loading: true };
    }
    const t = s.tracker;
    if (t && t.status === "centering") {
        const p = t.progress || 0;
        return { text: `${info.centering}... ${Math.round(p * 100)}%`, loading: true, progress: p };
    }
    if (t && t.status === "paused") {
        return { text: `Tracking: ${info.label} - paused (laugh / cough)`, loading: false };
    }
    if (t && t.status === "lost") {
        return { text: `Can't see ${info.subject}`, loading: false };
    }
    if (t && typeof t.dwell === "number" && t.dwell > 0.15) {
        return { text: `Tracking: ${info.label} - clicking...`, loading: true, progress: t.dwell };
    }
    return { text: `Tracking: ${info.label}`, loading: false };
}

function pollState() {
    chrome.runtime.sendMessage({ action: "get_state" }, (s) => {
        if (chrome.runtime.lastError) return;
        if (!s || s.error === true) {
            setStatus("WGI server offline - start the WGI app");
            setLoading(false);
            pendingMode = null;
            updateButtonStates(false);
            btnRecenter.classList.add('hidden-el');
            return;
        }
        lastState = s;
        // While a start/switch is in flight the server may still report the
        // previous session; wait for the requested mode (or an error).
        if (pendingMode && ((s.is_tracking && s.mode === pendingMode) || s.model_status === "error")) {
            pendingMode = null;
        }
        if (pendingMode) return;
        if (s.is_tracking && s.mode && s.mode !== selectedMode) selectMode(s.mode);

        const d = describeState(s);
        setStatus(d.text);
        setLoading(d.loading, d.progress === undefined ? null : d.progress);
        updateButtonStates(!!s.is_tracking);
        btnRecenter.classList.toggle('hidden-el', !(s.is_tracking && s.mode && s.mode !== 'hands'));
        btnRecenter.disabled = !s.camera_ready;
    });
}

modeButtons.forEach(btn => {
    btn.addEventListener('click', () => {
        const mode = btn.dataset.mode;
        if (mode === selectedMode) return;
        selectMode(mode);
        // Switching while tracking swaps the model live; the server releases
        // the old one before loading the new one.
        if (lastState && lastState.is_tracking) startTracking(mode);
    });
});

btnStart.addEventListener('click', () => startTracking(selectedMode));

btnStop.addEventListener('click', () => {
    pendingMode = null;
    chrome.runtime.sendMessage({ action: "stop_tracking" });
    setStatus("Stopping...");
    setLoading(false);
    updateButtonStates(false);
});

btnRecenter.addEventListener('click', () => {
    chrome.runtime.sendMessage({ action: "recenter" });
});

chrome.runtime.onMessage.addListener((msg) => {
    if (msg && msg.error && msg.message) setStatus(msg.message);
});

renderMode();
updateButtonStates(false);
pollState();
setInterval(pollState, 400);

const root = document.documentElement;
const mainPanel = document.getElementById('main-panel');
const wizardPanel = document.getElementById('wizard-panel');
const steps = document.querySelectorAll('.wizard-step');
let currentStep = 0;
let feedbackInterval;
let activeCalibrationGesture = null;
let advanceTimeout = null;

document.getElementById('theme-btn').addEventListener('click', () => {
    root.dataset.theme = root.dataset.theme === 'dark' ? 'light' : 'dark';
});

document.getElementById('btn-calibrate').addEventListener('click', () => {
    openWizard();
});

function resetWizardUi() {
    document.querySelectorAll('.btn-calibrate-action').forEach(button => {
        const label = getGestureLabel(button.dataset.gesture)
            .split(' ')
            .map(word => word.charAt(0).toUpperCase() + word.slice(1))
            .join(' ');
        button.disabled = false;
        button.innerText = `Calibrate ${label}`;
    });

    document.querySelectorAll('.saved-values').forEach(value => {
        value.classList.remove('visible');
        value.innerText = "";
    });

    document.querySelectorAll('.live-feedback[data-target]').forEach(fb => {
        setFeedback(fb, "Ready. Press calibrate, then hold the pose.", null);
        delete fb.dataset.lockUntil;
    });

    document.querySelectorAll('.wizard-step').forEach(step => {
        delete step.dataset.completed;
    });

    const nextButtons = document.querySelectorAll('.wizard-step .btn-next');
    if (nextButtons[1]) nextButtons[1].innerText = "Next: Right Click";
    if (nextButtons[2]) nextButtons[2].innerText = "Next: Double Click";
    if (nextButtons[3]) nextButtons[3].innerText = "Finish Setup";
}

function openWizard() {
    mainPanel.classList.add('hidden');
    wizardPanel.classList.remove('hidden');
    resetWizardUi();

    chrome.runtime.sendMessage({ action: "set_wizard_mode", active: true });

    chrome.runtime.sendMessage({ action: "get_base_url" }, (res) => {
        if (res && res.base_url) {
            // Ask for the stream at the size we are actually going to paint it,
            // in real device pixels, so it stays sharp on HiDPI displays instead
            // of being pinned to a resolution chosen on the server.
            const img = document.getElementById('video-feed');
            const dpr = window.devicePixelRatio || 1;
            const cssWidth = img.getBoundingClientRect().width || img.clientWidth || 320;
            const width = Math.max(160, Math.min(1280, Math.round(cssWidth * dpr)));
            img.src = `${res.base_url}/video_feed?width=${width}&t=${Date.now()}`;
        }
    });

    showStep(0);
}

function closeWizard() {
    wizardPanel.classList.add('hidden');
    mainPanel.classList.remove('hidden');
    document.getElementById('video-feed').src = "";
    currentStep = 0;
    activeCalibrationGesture = null;
    clearInterval(feedbackInterval);
    clearTimeout(advanceTimeout);
    chrome.runtime.sendMessage({ action: "set_wizard_mode", active: false });
}

function showStep(index) {
    steps.forEach((step, i) => step.classList.toggle('active', i === index));
    clearTimeout(advanceTimeout);
    if (steps[index]?.querySelector('.btn-calibrate-action')) {
        startLiveFeedback();
    } else {
        clearInterval(feedbackInterval);
    }
}

document.querySelectorAll('.btn-next').forEach(btn => {
    btn.addEventListener('click', () => {
        if (currentStep < steps.length - 1) {
            currentStep++;
            showStep(currentStep);
        } else {
            closeWizard();
        }
    });
});

document.querySelectorAll('.btn-calibrate-action').forEach(btn => {
    btn.addEventListener('click', (e) => {
        startCalibrationCapture(e.target.dataset.gesture, e.target);
    });
});

function setFeedback(fb, message, state, lockMs = 0) {
    fb.classList.remove('success', 'warning', 'error');
    if (state) fb.classList.add(state);
    fb.innerText = message;
    if (lockMs > 0) {
        fb.dataset.lockUntil = String(Date.now() + lockMs);
    } else {
        delete fb.dataset.lockUntil;
    }
}

function setStepComplete(step, gesture) {
    step.dataset.completed = "true";

    const button = step.querySelector('.btn-calibrate-action');
    if (button) {
        button.disabled = true;
        button.innerText = "OK - Calibrated";
    }

    const nextButton = step.querySelector('.btn-next');
    if (nextButton) {
        nextButton.innerText = currentStep < steps.length - 2 ? "Continuing..." : "Completing...";
    }

    advanceTimeout = setTimeout(() => {
        if (currentStep < steps.length - 1) {
            currentStep++;
            showStep(currentStep);
        } else {
            closeWizard();
        }
    }, 1200);
}

function getGestureLabel(gesture) {
    return gesture.replace('_', ' ');
}

function formatSavedValues(gesture, config) {
    const values = config && config[gesture];
    if (!values) return "Saved, but the values could not be loaded.";

    const usefulKeys = {
        left_click: ["idx_angle_max", "mid_angle_min", "thumb_index_dist_min"],
        right_click: ["idx_angle_min", "mid_angle_max", "thumb_index_dist_min"],
        double_click: ["trigger_angle_max", "release_angle_min", "thumb_index_dist_min"]
    }[gesture] || Object.keys(values);

    return "Saved: " + usefulKeys
        .map(key => `${key}=${Number(values[key]).toFixed(1)}`)
        .join(", ");
}

function showSavedValues(gesture) {
    chrome.runtime.sendMessage({ action: "get_config" }, (config) => {
        const target = document.querySelector(`[data-values-for="${gesture}"]`);
        if (!target) return;

        target.innerText = formatSavedValues(gesture, config);
        target.classList.add('visible');
    });
}

function startCalibrationCapture(gesture, button) {
    const step = button.closest('.wizard-step');
    const fb = step.querySelector('.live-feedback');
    const saved = step.querySelector('.saved-values');

    chrome.runtime.sendMessage({ action: "get_state" }, (state) => {
        if (!state || state.error || !state.is_tracking) {
            setFeedback(fb, "Start Camera first, then run calibration.", 'error', 1800);
            return;
        }

        activeCalibrationGesture = gesture;
        button.disabled = true;
        if (saved) {
            saved.classList.remove('visible');
            saved.innerText = "";
        }
        setFeedback(fb, `Recording ${getGestureLabel(gesture)}. Hold still...`, 'warning');

        chrome.runtime.sendMessage({ action: "start_calibration", gesture: gesture }, (res) => {
            if (!res || res.error) {
                button.disabled = false;
                activeCalibrationGesture = null;
                setFeedback(fb, res?.message || "Could not start calibration. Is the WGI server running?", 'error');
            }
        });
    });
}

function startLiveFeedback() {
    clearInterval(feedbackInterval);
    feedbackInterval = setInterval(() => {
        chrome.runtime.sendMessage({ action: "get_state" }, (res) => {
            if (!res || !res.action) return;

            const step = steps[currentStep];
            const fb = step?.querySelector('.live-feedback');
            const button = step?.querySelector('.btn-calibrate-action');
            if (!fb) return;
            const calibration = res.calibration || {};
            const calibrationStatus = calibration.status;
            const calibrationGesture = calibration.gesture;

            if (calibrationStatus === "recording" && calibrationGesture === activeCalibrationGesture) {
                setFeedback(fb, "Recording. Keep the exact same pose...", 'warning');
                return;
            }

            if (calibrationStatus === "saved" && calibrationGesture === activeCalibrationGesture) {
                const gesture = activeCalibrationGesture;
                if (button) button.disabled = false;
                activeCalibrationGesture = null;
                setFeedback(fb, `OK - ${getGestureLabel(gesture)} calibrated.`, 'success');
                if (gesture) showSavedValues(gesture);
                setStepComplete(step, gesture);
                return;
            }

            if (calibrationStatus === "failed" && calibrationGesture === activeCalibrationGesture) {
                if (button) button.disabled = false;
                activeCalibrationGesture = null;
                setFeedback(fb, "Calibration rejected. Use the exact pose shown and try again.", 'error', 1800);
                return;
            }

            if (activeCalibrationGesture) return;
            if (step.dataset.completed === "true") return;
            if (Number(fb.dataset.lockUntil || 0) > Date.now()) return;

            const targetAction = fb.dataset.target;
            if (res.action === targetAction || (targetAction === 'left_click' && res.action === 'left_click_intent')) {
                setFeedback(fb, "Pose detected. You can recalibrate if it feels off.", 'success');
            } else {
                setFeedback(fb, "Ready. Press calibrate, then hold the pose.", null);
            }
        });
    }, 150);
}
