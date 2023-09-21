document.addEventListener('DOMContentLoaded', function() {
    console.log("DOM fully loaded and parsed");  // Log when DOM is ready
    
    const chatBox = document.getElementById('bot-message');

    function appendMessage(message, className, isBotResponse = false) {
        const messageDiv = document.createElement('div');
        messageDiv.classList.add('message', className);
        
        const pre = document.createElement('pre');
    
        if (isBotResponse) {
            const formattedMessage = `${message.title} ${message.script}`;
            pre.textContent = formattedMessage;
        } else {
            pre.textContent = message;
        }
    
        // Style words that end with ":"
        styleColonWords(pre);
    
        messageDiv.appendChild(pre);
        document.getElementById('chat-box').appendChild(messageDiv);
    }
    
    function styleColonWords(element) {
        const text = element.textContent;
        const styledText = text.replace(/([\w\s\(\)]+):/g, '<span style="text-decoration: underline; font-weight: bold;">$1:</span>');
        element.innerHTML = styledText;
    }
    
async function fetchResponseFromBackend(userMessage) {
        console.log(`Fetching response for message: ${userMessage}`);  // Log the user message
        const response = await fetch('/', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ prompt: userMessage })
        });
    
        if (response.ok) {
            const data = await response.json();
            console.log("Frontend received:", data);  // Log the received data
            return data;
        } else {
            console.log("An error occurred while fetching");  // Log error during fetch
            return 'An error occurred';
        }
    } 

    const sendButton = document.getElementById('send-button');
    const userInput = document.getElementById('user-input');
        
    sendButton.addEventListener('click', async function() {
        console.log("Send button clicked");  // Log button click event
        const userMessage = userInput.value.trim();
    
        if (userMessage.length > 0) {  // Changed from 0.9 to 0
            console.log(`User message is: ${userMessage}`);  // Log the user's message
            appendMessage(userMessage, 'user-message');
            userInput.value = '';
    
            const botData = await fetchResponseFromBackend(userMessage);
    
            if (botData.error) {
                appendMessage(botData.error, 'bot-message');
            } else {
                console.log("Bot returned a valid response");
                appendMessage(botData, 'bot-message', true);
            }
        }
    });
});
