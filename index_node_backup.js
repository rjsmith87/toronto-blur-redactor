const express = require("express");

const app = express();
app.use(express.json({ limit: "10mb" }));

const port = process.env.PORT || 3000;

app.get("/health", (req, res) => {
  res.status(200).json({ status: "ok" });
});

// Stub redaction endpoint (we'll wire Google AI Studio next)
app.post("/redact", async (req, res) => {
  // Expect either a URL or base64 image data (we'll standardize later)
  const { imageUrl, imageBase64, metadata } = req.body || {};
  res.status(200).json({
    redacted: true,
    method: imageUrl ? "url" : imageBase64 ? "base64" : "none",
    receivedMetadataKeys: metadata ? Object.keys(metadata) : [],
    note: "stub response - Google AI Studio not wired yet"
  });
});

app.listen(port, () => {
  console.log(`heroku-redactor listening on port ${port}`);
});
