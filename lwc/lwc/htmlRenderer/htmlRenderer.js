import { LightningElement, api, wire } from 'lwc';
import { getRecord, getFieldValue } from 'lightning/uiRecordApi';

export default class HtmlRenderer extends LightningElement {
    @api recordId;
    @api objectApiName;
    @api fieldApiName;

    htmlContent;
    error;
    renderedOnce = false;

    get dynamicField() {
        if (this.objectApiName && this.fieldApiName) {
            return `${this.objectApiName}.${this.fieldApiName}`;
        }
        return null;
    }

    get dynamicFieldsArray() {
        return this.dynamicField ? [this.dynamicField] : [];
    }

    @wire(getRecord, { recordId: '$recordId', fields: '$dynamicFieldsArray' })
    wiredRecord({ error, data }) {
        if (!this.dynamicField) {
            this.error = 'Component is not configured. Please provide both objectApiName and fieldApiName.';
            this.htmlContent = undefined;
            if (this.renderedOnce) {
                this.clearHtmlContainer();
            }
            return;
        }
        if (data) {
            this.htmlContent = getFieldValue(data, this.dynamicField);
            this.error = undefined;
            if (this.htmlContent === undefined) {
                // Optionally add more user guidance here
            } else if (this.htmlContent === null || (typeof this.htmlContent === 'string' && this.htmlContent.trim() === "")) {
                // Optionally handle empty
            }
            if (this.renderedOnce && this.htmlContent) {
                this.renderHtml();
            }
        } else if (error) {
            this.error = error;
            this.htmlContent = undefined;
            if (this.renderedOnce) {
                this.clearHtmlContainer();
            }
        }
    }

    renderedCallback() {
        if (this.htmlContent && !this.renderedOnce) {
            this.renderHtml();
            this.renderedOnce = true;
        } else if (!this.htmlContent && this.renderedOnce) {
            this.clearHtmlContainer();
            this.renderedOnce = false; 
        } else if (this.htmlContent && this.renderedOnce) {
            const container = this.template.querySelector('.html-container');
            if (container && container.innerHTML !== this.htmlContent) {
                this.renderHtml();
            }
        }
    }

    renderHtml() {
        const wrapper = this.template.querySelector('.html-renderer-wrapper');
        const container = this.template.querySelector('.html-container');
        
        if (container && this.htmlContent && wrapper) {
            // Extract style tag content and inject it separately
            const styleMatch = this.htmlContent.match(/<style[^>]*>([\s\S]*?)<\/style>/i);
            let htmlWithoutStyle = this.htmlContent;
            
            // Remove existing style element if present
            const existingStyle = wrapper.querySelector('style.html-renderer-styles');
            if (existingStyle) {
                existingStyle.remove();
            }
            
            if (styleMatch) {
                // Remove style tag from HTML content
                htmlWithoutStyle = this.htmlContent.replace(/<style[^>]*>[\s\S]*?<\/style>/gi, '');
                
                // Create and inject style element
                const styleElement = document.createElement('style');
                styleElement.className = 'html-renderer-styles';
                styleElement.type = 'text/css';
                styleElement.textContent = styleMatch[1];
                
                // Insert style before the container
                wrapper.insertBefore(styleElement, container);
            }
            
            // Set HTML content (without style tag)
            container.innerHTML = htmlWithoutStyle;
            
            // Handle external links - add target and rel attributes for security
            const links = container.querySelectorAll('a[href^="http"]');
            links.forEach(link => {
                if (!link.hasAttribute('target')) {
                    link.setAttribute('target', '_blank');
                }
                if (!link.hasAttribute('rel')) {
                    link.setAttribute('rel', 'noopener noreferrer');
                }
            });
        }
    }

    clearHtmlContainer() {
        const wrapper = this.template.querySelector('.html-renderer-wrapper');
        const container = this.template.querySelector('.html-container');
        
        if (container) {
            container.innerHTML = '';
        }
        // Remove injected styles
        if (wrapper) {
            const styleElement = wrapper.querySelector('style.html-renderer-styles');
            if (styleElement) {
                styleElement.remove();
            }
        }
    }

    get hasHtmlContent() {
        if (typeof this.htmlContent === 'string') {
            return this.htmlContent.trim().length > 0;
        }
        return !!this.htmlContent;
    }

    get errorMessage() {
        if (this.error) {
            if (this.error.body && this.error.body.message) {
                if (this.error.body.output && this.error.body.output.errors && this.error.body.output.errors.length > 0) {
                    const detailedError = this.error.body.output.errors[0].message;
                    return `Error fetching data: ${this.error.body.message} (Details: ${detailedError})`;
                }
                return `Error fetching data: ${this.error.body.message}`;
            }
            return typeof this.error === 'string' ? this.error : 'An unknown error occurred while fetching data.';
        }
        return null;
    }
}