export const PROVIDERS = {
  vllm: {
    apiBaseUrl: "http://localhost:8000/v1",
    apiKey: "vllm",
    apiType: ["responses", "chat"], // choose from responses, chat, or both
    modelName: "openai/gpt-oss-120b",
    providerDetails: {
      // add any provider-specific details here. These will be passed as part of every request
      // for example to fix the provider for openrouter, you can do:
      // provider: {
      //   only: ["example"],
      // },
    },
  },
  openai: {
    apiBaseUrl: "https://api.openai.com/v1",
    apiKey: process.env.OPENAI_API_KEY ?? "",
    apiType: ["responses"], // choose from responses, chat, or both
    modelName: "gpt-5",
    providerDetails: {
      // add any provider-specific details here. These will be passed as part of every request
      // for example to fix the provider for openrouter, you can do:
      // provider: {
      //   only: ["example"],
      // },
    },
  },
};
