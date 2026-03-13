// ─── CONFIG ───────────────────────────────────────────────
const TARGET_SENTENCE = "the quick brown fox jumps over the lazy dog near the river bank";
const TOTAL_ATTEMPTS  = 15;

// ─── STATE ────────────────────────────────────────────────
let username      = "";
let attemptNumber = 1;
let allAttempts   = [];   // stores all 15 attempts
let keyLog        = [];   // raw events for current attempt
let isRecording   = false;

// ─── INIT ─────────────────────────────────────────────────
document.getElementById("target-sentence").textContent = TARGET_SENTENCE;

// ─── STEP 1: Start session ────────────────────────────────
function startSession() {
    const input = document.getElementById("username").value.trim();
    if (!input) {
        alert("Please enter your name first.");
        return;
    }
    username = input;
    document.getElementById("step-name").style.display   = "none";
    document.getElementById("step-typing").style.display = "block";
    startAttempt();
}

// ─── STEP 2: Each attempt ─────────────────────────────────
function startAttempt() {
    keyLog      = [];
    isRecording = true;

    const area = document.getElementById("typing-area");
    area.value = "";
    area.focus();

    document.getElementById("attempt-counter").textContent =
        `Attempt: ${attemptNumber} / ${TOTAL_ATTEMPTS}`;
    document.getElementById("status-msg").textContent = "";
}

// ─── KEYSTROKE CAPTURE ────────────────────────────────────
const typingArea = document.getElementById("typing-area");

typingArea.addEventListener("keydown", (e) => {
    if (!isRecording) return;
    keyLog.push({
        key:   e.key,
        event: "down",
        time:  performance.now()   // millisecond precision
    });
});

typingArea.addEventListener("keyup", (e) => {
    if (!isRecording) return;
    keyLog.push({
        key:   e.key,
        event: "up",
        time:  performance.now()
    });
});

// ─── SUBMIT ONE ATTEMPT ───────────────────────────────────
function submitAttempt() {
    const typed = document.getElementById("typing-area").value.trim();

    // Basic validation
    if (typed.length < 10) {
        document.getElementById("status-msg").textContent =
            "⚠️ Please type the full sentence before submitting.";
        return;
    }

    isRecording = false;

    // Process raw keylog into features
    const features = extractFeatures(keyLog);

    allAttempts.push({
        attempt:  attemptNumber,
        typed:    typed,
        raw_log:  keyLog,
        features: features
    });

    if (attemptNumber < TOTAL_ATTEMPTS) {
        attemptNumber++;
        document.getElementById("status-msg").textContent =
            "✅ Saved! Get ready for the next attempt...";
        setTimeout(startAttempt, 1000);
    } else {
        // All done
        document.getElementById("step-typing").style.display = "none";
        document.getElementById("step-done").style.display   = "block";
    }
}

// ─── FEATURE EXTRACTION ───────────────────────────────────
function extractFeatures(log) {
    const keydowns = {};   // key → last keydown time
    const dwellTimes  = {};   // key → list of dwell times
    const flightTimes = [];   // list of {from, to, time}

    let lastUpKey  = null;
    let lastUpTime = null;

    log.forEach(event => {
        if (event.event === "down") {
            keydowns[event.key] = event.time;

            // Flight time: from last keyup to this keydown
            if (lastUpTime !== null) {
                flightTimes.push({
                    from: lastUpKey,
                    to:   event.key,
                    time: event.time - lastUpTime
                });
            }
        }

        if (event.event === "up") {
            // Dwell time: how long was this key held
            if (keydowns[event.key] !== undefined) {
                const dwell = event.time - keydowns[event.key];
                if (!dwellTimes[event.key]) dwellTimes[event.key] = [];
                dwellTimes[event.key].push(dwell);
            }

            lastUpKey  = event.key;
            lastUpTime = event.time;
        }
    });

    // Summarize dwell times per key (mean)
    const dwellMeans = {};
    for (const key in dwellTimes) {
        const times = dwellTimes[key];
        dwellMeans[key] = times.reduce((a, b) => a + b, 0) / times.length;
    }

    // Overall stats
    const allDwells  = Object.values(dwellTimes).flat();
    const allFlights = flightTimes.map(f => f.time);

    return {
        dwell_per_key:     dwellMeans,
        flight_times:      flightTimes,
        mean_dwell:        mean(allDwells),
        std_dwell:         std(allDwells),
        mean_flight:       mean(allFlights),
        std_flight:        std(allFlights),
        total_keys:        log.filter(e => e.event === "down").length,
        backspace_count:   log.filter(e => e.event === "down" && e.key === "Backspace").length,
        total_time_ms:     log.length > 1 ? log[log.length-1].time - log[0].time : 0
    };
}

// ─── MATH HELPERS ─────────────────────────────────────────
function mean(arr) {
    if (!arr.length) return 0;
    return arr.reduce((a, b) => a + b, 0) / arr.length;
}

function std(arr) {
    if (arr.length < 2) return 0;
    const m = mean(arr);
    return Math.sqrt(arr.reduce((sum, x) => sum + (x - m) ** 2, 0) / arr.length);
}

// ─── DOWNLOAD DATA ────────────────────────────────────────
function downloadData() {
    const payload = {
        username:   username,
        sentence:   TARGET_SENTENCE,
        collected:  new Date().toISOString(),
        attempts:   allAttempts
    };

    const blob = new Blob([JSON.stringify(payload, null, 2)], 
                          { type: "application/json" });
    const url  = URL.createObjectURL(blob);
    const a    = document.createElement("a");
    a.href     = url;
    a.download = `${username}_keystrokes.json`;
    a.click();
    URL.revokeObjectURL(url);
}