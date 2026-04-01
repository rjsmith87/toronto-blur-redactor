import { LightningElement, api, wire } from 'lwc';
import { getRecord, getFieldValue } from 'lightning/uiRecordApi';
import { loadScript } from 'lightning/platformResourceLoader';
// Ensure this matches your Static Resource Name
import MARKED_JS from '@salesforce/resourceUrl/marked_js';

export default class MarkdownViewer extends LightningElement {
    @api recordId;
    @api objectApiName;
    @api fieldApiName;

    error;
    renderedHtml;
    isLoading = true;
    markdownText;
    scriptLoaded = false;

    get dynamicField() {
        if (this.objectApiName && this.fieldApiName) {
            return `${this.objectApiName}.${this.fieldApiName}`;
        }
        return null;
    }

    @wire(getRecord, {
        recordId: '$recordId',
        fields: '$dynamicFieldsArray'
    })
    wiredRecord({ error, data }) {
        if (!this.dynamicField) {
            this.error = 'Component is not configured. Please provide both objectApiName and fieldApiName.';
            this.markdownText = undefined;
            this.renderedHtml = undefined;
            this.isLoading = false;
            return;
        }
        if (data) {
            const fieldValue = getFieldValue(data, this.dynamicField);
            this.markdownText = fieldValue;
            this.error = undefined;
            if (this.scriptLoaded) {
                this.renderMarkdown();
            }
        } else if (error) {
            this.error = 'Failed to load record data. See console.';
            this.markdownText = undefined;
            this.renderedHtml = undefined;
            this.isLoading = false;
        }
    }

    get dynamicFieldsArray() {
        return this.dynamicField ? [this.dynamicField] : [];
    }

    renderedCallback() {
        if (this.scriptLoaded) {
            return;
        }
        loadScript(this, MARKED_JS)
            .then(() => {
                this.scriptLoaded = true;
                this.error = undefined;
                if (this.markdownText !== undefined) {
                    this.renderMarkdown();
                }
            })
            .catch(error => {
                this.error = 'Failed to load Markdown library. See console.';
                this.renderedHtml = undefined;
                this.isLoading = false;
            });
    }

    renderMarkdown() {
        const markedVar = typeof marked !== 'undefined' ? marked : (typeof window !== 'undefined' ? window.marked : undefined);
        const markedType = typeof markedVar;
        let finalHtml = undefined;
        let hadError = false;
        let markedFunction = undefined;
        if (markedType === 'function') {
            markedFunction = markedVar;
        } else if (markedType === 'object' && markedVar !== null) {
            if (typeof markedVar.marked === 'function') {
                markedFunction = markedVar.marked;
            } else if (typeof markedVar.parse === 'function') {
                markedFunction = markedVar.parse;
            } else if (typeof markedVar.default === 'function') {
                markedFunction = markedVar.default;
            }
        }
        if (this.scriptLoaded && this.markdownText !== undefined && this.markdownText !== null && typeof this.markdownText === 'string' && this.markdownText.trim() !== '' && typeof markedFunction === 'function') {
            try {
                const htmlResult = markedFunction(this.markdownText);
                finalHtml = htmlResult;
            } catch (e) {
                this.error = 'Failed to render Markdown content. See console.';
                finalHtml = undefined;
                hadError = true;
            }
        } else if (this.scriptLoaded && (this.markdownText === null || (typeof this.markdownText === 'string' && this.markdownText.trim() === ''))) {
            finalHtml = undefined;
        } else if (this.scriptLoaded && this.markdownText === undefined) {
            this.isLoading = true;
            return;
        } else if (!this.scriptLoaded) {
            this.isLoading = true;
            return;
        } else {
            this.error = 'Markdown library loaded as object, but failed to find callable function inside (checked .marked, .parse, .default).';
            finalHtml = undefined;
            hadError = true;
        }
        this.renderedHtml = finalHtml;
        if (!hadError) {
            this.error = undefined;
        }
        this.isLoading = false;
    }
}