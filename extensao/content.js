const feedbackToast = document.createElement("div");
feedbackToast.style.position = "fixed";
feedbackToast.style.bottom = "20px";
feedbackToast.style.right = "20px";
feedbackToast.style.padding = "9px 18px";
feedbackToast.style.background = "linear-gradient(180deg, rgba(158,236,255,0.95) 0%, rgba(31,178,236,0.95) 50%, rgba(11,127,196,0.95) 51%, rgba(31,178,236,0.95) 100%)";
feedbackToast.style.border = "1px solid rgba(255,255,255,0.9)";
feedbackToast.style.boxShadow = "inset 0 1px 0 rgba(255,255,255,0.8), 0 6px 18px rgba(11,127,196,0.45)";
feedbackToast.style.textShadow = "0 1px 1px rgba(0,60,100,0.55)";
feedbackToast.style.fontWeight = "600";
feedbackToast.style.color = "#ffffff";
feedbackToast.style.borderRadius = "999px";
feedbackToast.style.fontFamily = "\"Segoe UI\", \"Frutiger\", \"Myriad Pro\", system-ui, sans-serif";
feedbackToast.style.fontSize = "14px";
feedbackToast.style.zIndex = "999999";
feedbackToast.style.opacity = "0";
feedbackToast.style.transform = "translateY(20px)";
feedbackToast.style.transition = "opacity 0.3s ease, transform 0.3s cubic-bezier(0.175, 0.885, 0.32, 1.275)";
feedbackToast.style.pointerEvents = "none";
document.body.appendChild(feedbackToast);

let feedbackTimeout;

const friendlyNames = {
    "left_click": "Left click",
    "right_click": "Right click",
    "double_click": "Double click",
    "scroll": "Scrolling",
    "drag": "Dragging",
    "release_drag": "Dropped"
};

// Status-only actions: they repeat every frame and must not raise a toast.
const ignoredActions = [
    "move", "holding_click", "left_click_intent",
    "centering", "eyes_closed", "scroll_ready", "shrug",
    "wink_left", "wink_right", "paused", "learning", "pinch",
    "calibrating"
];

const intervalId = setInterval(() => {
    try {
        chrome.runtime.sendMessage({ action: "get_state" }, (response) => {
            if (chrome.runtime.lastError) {
                return;
            }
            if (response && !ignoredActions.includes(response.action)) {
                console.log("Ação detectada pelo WGI:", response.action);
                
                const actionText = friendlyNames[response.action] || response.action;
                feedbackToast.innerText = actionText;
                feedbackToast.style.opacity = "1";
                feedbackToast.style.transform = "translateY(0)";
                
                // Hide it again after 1 second
                clearTimeout(feedbackTimeout);
                feedbackTimeout = setTimeout(() => {
                    feedbackToast.style.opacity = "0";
                    feedbackToast.style.transform = "translateY(20px)";
                }, 1000);
            }
        });
    } catch (error) {
        if (error.message.includes("Extension context invalidated")) {
            clearInterval(intervalId);
            feedbackToast.remove();
        }
    }
}, 50);