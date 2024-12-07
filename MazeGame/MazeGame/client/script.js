document.addEventListener("DOMContentLoaded", function () {
    const SERVER_URL = "https://dory-unbiased-marlin.ngrok-free.app//api/motion-data";

    let motionData = [];
    const canvas = document.getElementById("gameCanvas");
    const ctx = canvas.getContext("2d");
    const restartButton = document.getElementById("restartGame");
    const messageElement = document.getElementById("message");
    const upButton = document.getElementById("up");
    const downButton = document.getElementById("down");
    const leftButton = document.getElementById("left");
    const rightButton = document.getElementById("right");
    const playSineWaveButton = document.getElementById("playSineWave");
    const stopSineWaveButton = document.getElementById("stopSineWave");

    let player = { x: 40, y: 40, size: 40, color: "red" };
    const blockSize = canvas.width / 10;
    const maze = [
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 0, 0, 0, 1, 0, 0, 0, 0, 1],
        [1, 0, 1, 0, 1, 0, 1, 1, 0, 1],
        [1, 0, 1, 0, 0, 0, 1, 0, 0, 1],
        [1, 0, 1, 1, 1, 0, 1, 0, 1, 1],
        [1, 0, 0, 0, 1, 0, 0, 0, 0, 1],
        [1, 1, 1, 0, 1, 1, 1, 1, 0, 1],
        [1, 0, 0, 0, 0, 0, 0, 1, 0, 1],
        [1, 0, 1, 1, 1, 1, 0, 1, 0, 0],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    ];

    let sineWaveInterval;
    let oscillator;
    let audioContext;

    function handleMotionEvent(event) {
        const { acceleration, rotationRate } = event;
        const data = {
            timestamp: Date.now(),
            acceleration: {
                x: acceleration.x || 0,
                y: acceleration.y || 0,
                z: acceleration.z || 0
            },
            rotationRate: {
                alpha: rotationRate.alpha || 0,
                beta: rotationRate.beta || 0,
                gamma: rotationRate.gamma || 0
            }
        };
        motionData.push(data);
        if (motionData.length > 50) motionData.shift();
    }

    function sendMotionData() {
        if (motionData.length > 0) {
            fetch(SERVER_URL, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ motionData })
            })
                .then(response => response.json())
                .then(data => console.log("Data sent to server:", data))
                .catch(error => console.error("Error sending data:", error));

            motionData = [];
        }
    }

    setInterval(sendMotionData, 3000);

    function startMotionCapture() {
        if (window.DeviceMotionEvent) {
            window.addEventListener("devicemotion", handleMotionEvent);
        } else {
            console.warn("DeviceMotionEvent is not supported on this device.");
        }
    }

    function startGame() {
        drawMaze();
        drawPlayer();
        messageElement.textContent = "화면의 미로에서 캐릭터를 움직여 탈출하세요!";
    }

    function drawMaze() {
        for (let row = 0; row < maze.length; row++) {
            for (let col = 0; col < maze[row].length; col++) {
                if (maze[row][col] === 1) {
                    ctx.fillStyle = "#000";
                    ctx.fillRect(col * blockSize, row * blockSize, blockSize, blockSize);
                }
            }
        }
    }

    function drawPlayer() {
        ctx.fillStyle = player.color;
        ctx.fillRect(player.x, player.y, player.size, player.size);
    }

    function movePlayer(direction) {
        let newX = player.x;
        let newY = player.y;
        if (direction === "up") {
            newY -= blockSize;
        } else if (direction === "down") {
            newY += blockSize;
        } else if (direction === "left") {
            newX -= blockSize;
        } else if (direction === "right") {
            newX += blockSize;
        }

        const row = Math.floor(newY / blockSize);
        const col = Math.floor(newX / blockSize);
        if (maze[row] && maze[row][col] === 0) {
            player.x = newX;
            player.y = newY;
            ctx.clearRect(0, 0, canvas.width, canvas.height);
            drawMaze();
            drawPlayer();
        }

        if (row === maze.length - 1 && col === maze[0].length - 1) {
            messageElement.textContent = "축하합니다! 미로를 탈출하셨습니다!";
        }
    }

    // 키보드 이벤트 처리
    document.addEventListener("keydown", (event) => {
        const key = event.key;
        if (key === "ArrowUp") movePlayer("up");
        else if (key === "ArrowDown") movePlayer("down");
        else if (key === "ArrowLeft") movePlayer("left");
        else if (key === "ArrowRight") movePlayer("right");
    });

    // 버튼 클릭 이벤트 처리
    upButton.addEventListener("click", () => movePlayer("up"));
    downButton.addEventListener("click", () => movePlayer("down"));
    leftButton.addEventListener("click", () => movePlayer("left"));
    rightButton.addEventListener("click", () => movePlayer("right"));

    restartButton.addEventListener("click", function () {
        player.x = 40;
        player.y = 40;
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        startGame();
    });

    function playSineWave() {
        if (audioContext && audioContext.state === 'suspended') {
            audioContext.resume();
        } else {
            audioContext = new (window.AudioContext || window.webkitAudioContext)();
        }
        oscillator = audioContext.createOscillator();
        oscillator.type = "sine"; // 사인파 형태
        oscillator.frequency.setValueAtTime(440, audioContext.currentTime); // 440Hz (기본 A음)
        oscillator.connect(audioContext.destination);
        oscillator.start();
        
        // 사인파를 1초마다 계속 방출
        sineWaveInterval = setInterval(() => {
            oscillator.start();
            setTimeout(() => {
                oscillator.stop();
            }, 1000); // 1초 동안 사인파 재생
        }, 1000); // 1초 간격으로 반복
    }

    function stopSineWave() {
        clearInterval(sineWaveInterval);
        if (oscillator) {
            oscillator.stop();
        }
    }

    playSineWaveButton.addEventListener("click", playSineWave);
    stopSineWaveButton.addEventListener("click", stopSineWave);

    startGame();
    startMotionCapture();
});
