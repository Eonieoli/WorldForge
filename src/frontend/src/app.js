const chatContainer = document.getElementById("chat-container");
const messageForm = document.getElementById("message-form");
const userInput = document.getElementById("user-input");

const BASE_URL = "https://backend-winter-wind-1715.fly.dev";
const backendUrl = "http://localhost:8000/chat";

// Create a message bubble
function createMessageBubble(content, sender = "user") {
  const wrapper = document.createElement("div");
  if (sender === "assistant") {
    wrapper.classList.add("mb-6", "flex", "items-start", "space-x-3");
  } else {
    wrapper.classList.add("mb-6", "flex", "flex-row-reverse", "space-x-3", "space-x-reverse");
  }

  // Avatar (프로필 디자인)
  const avatar = document.createElement("div");
  avatar.classList.add(
    "w-10",
    "h-10",
    "rounded-full",
    "flex-shrink-0",
    "flex",
    "items-center",
    "justify-center",
    "font-bold",
    "text-white"
  );

  // 이미지 엘리먼트 생성 및 추가
const image = document.createElement("img");
image.src = "https://thumb.mt.co.kr/06/2024/10/2024101020084787406_1.jpg"; // 이미지 URL을 여기에 입력
image.alt = "Avatar";
image.classList.add("w-full", "h-full", "object-cover", "rounded-full");

  // 프로필 디자인 (개별 적용)
  if (sender === "assistant") {
    avatar.classList.add("bg-gradient-to-br", "from-green-400", "to-green-600");
    avatar.appendChild(image)
  } else {
    avatar.classList.add("bg-gradient-to-br", "from-orange-500", "to-orange-700");
    avatar.textContent = "U";
  }

  // Bubble(말풍선 디자인 공통적용)
  const bubble = document.createElement("div");
  bubble.classList.add(
    "max-w-full",
    "md:max-w-2xl",
    "p-3",
    // "rounded-lg",
    "whitespace-pre-wrap",
    "leading-relaxed",
    "shadow-sm"
  );

  if (sender === "assistant") {
    bubble.classList.add("bg-orange-50", "text-gray-900", "rounded-tr-lg", "rounded-tl-lg", "rounded-br-lg");
  } else {
    bubble.classList.add("bg-orange-200", "text-gray-900", "rounded-tr-lg", "rounded-tl-lg", "rounded-bl-lg");
  }

  bubble.textContent = content;

  wrapper.appendChild(avatar);
  wrapper.appendChild(bubble);
  console.log(wrapper)
  return wrapper;
}

// Scroll to bottom
function scrollToBottom() {
  chatContainer.scrollTop = chatContainer.scrollHeight;
}


// Fetch assistant response from the selected backend endpoint
async function getAssistantResponse(userMessage) {
  const response = await fetch(`${backendUrl}`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({ message: userMessage }),
  });
  console.log(response)
  if (!response.ok) {
    throw new Error("Network response was not ok");
  }

  const data = await response.json();
  if (typeof data !== "object") {
    throw new Error("Invalid JSON response");
  }
  return data.reply;
}


// Handle form submission
messageForm.addEventListener("submit", async (e) => {
  e.preventDefault();
  /////////// 여기에 답변 메세지를 담자
  const message = userInput.value.trim();
  if (!message) return;

  // User message
  chatContainer.appendChild(createMessageBubble(message, "user"));
  userInput.value = "";
  scrollToBottom();

  // Assistant response
  const response = await getAssistantResponse(message);
  chatContainer.appendChild(createMessageBubble(response, "assistant"));
  scrollToBottom();
});
