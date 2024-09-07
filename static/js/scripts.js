$(document).ready(function () {
    // Smooth scrolling for sections
    $('a[href*="#"]').on('click', function (e) {
        e.preventDefault();
        $('html, body').animate({
            scrollTop: $($(this).attr('href')).offset().top
        }, 500, 'linear');
    });

    // Auto-scroll to the bottom of the chat and voice conversation
    function scrollToBottom(selector) {
        $(selector).animate({ scrollTop: $(selector)[0].scrollHeight }, 500);
    }

    $('#text-button').on('click', function () {
        scrollToBottom('#chat-output');
    });

    $('#start-button, #stop-button').on('click', function () {
        scrollToBottom('#voice-conversation-output');
    });

    // Update conversation every 5 seconds
    setInterval(function () {
        $.getJSON('/update-conversation', function (data) {
            var conversationHtml = '';
            data.forEach(function (item) {
                conversationHtml += '<p><strong>' + item[0] + ':</strong> ' + item[1] + '</p>';
            });
            $('#voice-conversation-output').html(conversationHtml);
            scrollToBottom('#voice-conversation-output');
        });
    }, 5000);
});
