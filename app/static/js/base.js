//jQuery is a JavaScript Library.
//jQuery greatly simplifies JavaScript programming.
//jQuery is easy to learn.

//The jQuery syntax is tailor-made for selecting HTML elements and performing some action on the element(s).
//Basic syntax is: $(selector).action()
//A $ sign to define/access jQuery
//A (selector) to "query (or find)" HTML elements
//A jQuery action() to be performed on the element(s)

//The Document Ready Event
//ready is to prevent any jQuery code from running before the document is finished loading
$(document).ready(function () {
    $('body').children().not('.wrapper').remove();
});

//If your website contains a lot of pages, and you want your jQuery functions to be easy to maintain,
//you can put your jQuery functions in a separate .js file.