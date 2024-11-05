document.addEventListener("DOMContentLoaded", function() {
    const canvas = document.getElementById("gameCanvas");
    const ctx = canvas.getContext("2d");
    const restartButton = document.getElementById("restartGame");
    const messageElement = document.getElementById("message");

    // 새로운 코드 시작 (가속도와 회전 데이터를 저장하는 부분)
    let accelerationData = [];
    let rotationData = [];

    if (window.DeviceMotionEvent) {
        window.addEventListener("devicemotion", handleMotion);
    } else {
        alert("이 기기는 DeviceMotionEvent를 지원하지 않습니다.");
    }

    function handleMotion(event) {
        const acceleration = event.accelerationIncludingGravity;
        const rotationRate = event.rotationRate;

        accelerationData.push({
            x: acceleration.x || 0,
            y: acceleration.y || 0,
            z: acceleration.z || 0,
            timestamp: new Date().toISOString()
        });

        if (rotationRate) {
            rotationData.push({
                alpha: rotationRate.alpha || 0,
                beta: rotationRate.beta || 0,
                gamma: rotationRate.gamma || 0,
                timestamp: new Date().toISOString()
            });
        }

        // 콘솔에 데이터 출력
        console.log("Acceleration:", accelerationData[accelerationData.length - 1]);
        console.log("Rotation:", rotationData[rotationData.length - 1]);

        // 화면에 데이터 표시
        document.getElementById("accelerationData").textContent = `Acceleration - X: ${acceleration.x.toFixed(2)}, Y: ${acceleration.y.toFixed(2)}, Z: ${acceleration.z.toFixed(2)}`;
        document.getElementById("rotationData").textContent = `Rotation - Alpha: ${rotationRate.alpha.toFixed(2)}, Beta: ${rotationRate.beta.toFixed(2)}, Gamma: ${rotationRate.gamma.toFixed(2)}`;
    }
    // 새로운 코드 끝

 
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

    function startGame() {
        document.addEventListener("keydown", movePlayer);
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

    function movePlayer(event) {
        let newX = player.x;
        let newY = player.y;
        if (event.key === "ArrowUp") {
            newY -= blockSize;
        } else if (event.key === "ArrowDown") {
            newY += blockSize;
        } else if (event.key === "ArrowLeft") {
            newX -= blockSize;
        } else if (event.key === "ArrowRight") {
            newX += blockSize;
        }

        // Check if the new position is within bounds and not a wall
        const row = Math.floor(newY / blockSize);
        const col = Math.floor(newX / blockSize);
        if (maze[row] && maze[row][col] === 0) {
            player.x = newX;
            player.y = newY;
            ctx.clearRect(0, 0, canvas.width, canvas.height);
            drawMaze();
            drawPlayer();
        }

        // Check if player has reached the exit
        if (row === maze.length - 1 && col === maze[0].length - 1) {
            messageElement.textContent = "축하합니다! 미로를 탈출하셨습니다!";
            document.removeEventListener("keydown", movePlayer); // 게임 종료 후 이동 방지
        }
    }

    restartButton.addEventListener("click", function() {
        player.x = 40;
        player.y = 40;
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        startGame();
    });

    startGame();
});