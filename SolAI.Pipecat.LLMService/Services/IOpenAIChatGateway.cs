using Microsoft.AspNetCore.Http;
using SolAI.Pipecat.LLMService.Contracts;

namespace SolAI.Pipecat.LLMService.Services;

public interface IOpenAIChatGateway
{
    OpenAIModelListResponse GetModels();

    Task<OpenAIChatCompletionResponse> CreateCompletionAsync(
        OpenAIChatCompletionRequest request,
        CancellationToken cancellationToken);

    Task WriteStreamingCompletionAsync(
        OpenAIChatCompletionRequest request,
        HttpResponse response,
        CancellationToken cancellationToken);
}
