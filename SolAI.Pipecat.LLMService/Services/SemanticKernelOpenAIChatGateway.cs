using System.Text;
using System.Diagnostics;

using System.Text.Json;
using System.Text.Json.Serialization;
using System.Text.RegularExpressions;
using Microsoft.AspNetCore.Http;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Options;
using Microsoft.SemanticKernel;
using Microsoft.SemanticKernel.ChatCompletion;
using Microsoft.SemanticKernel.Connectors.OpenAI;
using SolAI.Pipecat.LLMService.Contracts;
using SolAI.Pipecat.LLMService.Options;
using SolAI.Pipecat.LLMService.Plugins;

namespace SolAI.Pipecat.LLMService.Services;

public sealed class SemanticKernelOpenAIChatGateway : IOpenAIChatGateway
{
    private static readonly Regex ThinkBlockRegex = new(
        "<think>.*?</think>",
        RegexOptions.IgnoreCase | RegexOptions.Singleline | RegexOptions.Compiled);

    private static readonly JsonSerializerOptions SseSerializerOptions = new(JsonSerializerDefaults.Web)
    {
        DefaultIgnoreCondition = JsonIgnoreCondition.WhenWritingNull
    };

    public const string ActivitySourceName = "SolAI.Pipecat.LLMService";
    private static readonly ActivitySource ActivitySource = new(ActivitySourceName);


    private readonly LlmServiceOptions _options;
    private readonly ILogger<SemanticKernelOpenAIChatGateway> _logger;

    public SemanticKernelOpenAIChatGateway(
        IOptions<LlmServiceOptions> options,
        ILogger<SemanticKernelOpenAIChatGateway> logger)
    {
        _options = options.Value;
        _logger = logger;
    }

    public OpenAIModelListResponse GetModels()
    {
        var created = DateTimeOffset.UtcNow.ToUnixTimeSeconds();
        var models = _options.AvailableModels.Count == 0
            ? [_options.DefaultModel]
            : _options.AvailableModels.Distinct(StringComparer.OrdinalIgnoreCase).ToList();

        return new OpenAIModelListResponse
        {
            Data = models.Select(model => new OpenAIModelCard
            {
                Id = model,
                Created = created
            }).ToList()
        };
    }

    public async Task<OpenAIChatCompletionResponse> CreateCompletionAsync(
        OpenAIChatCompletionRequest request,
        CancellationToken cancellationToken)
    {
        using var activity = ActivitySource.StartActivity("llm.create_completion", ActivityKind.Internal);
        activity?.SetTag("llm.model", request.Model ?? _options.DefaultModel);
        activity?.SetTag("llm.stream", false);
        activity?.SetTag("llm.message_count", request.Messages.Count);
        activity?.SetTag("llm.tool_choice.kind", request.ToolChoice.ValueKind.ToString());

        var model = ResolveModel(request.Model);
        var kernel = BuildKernel(model);
        var chatHistory = BuildChatHistory(request);
        var chatService = kernel.GetRequiredService<IChatCompletionService>();
        var settings = BuildExecutionSettings(request, kernel);

        _logger.LogInformation(
            "Elaborazione chat completion con modello {Model}. Stream={Stream}, Messaggi={MessageCount}",
            model,
            request.Stream,
            request.Messages.Count);

        try
        {
            var result = await chatService.GetChatMessageContentAsync(
                chatHistory,
                settings,
                kernel,
                cancellationToken);

            var content = result.Content ?? result.ToString();
            activity?.SetTag("llm.response.length", content?.Length ?? 0);
            activity?.SetStatus(ActivityStatusCode.Ok);
            return BuildCompletionResponse(model, SanitizeAssistantContent(content ?? string.Empty));
        }
        catch (OperationCanceledException)
        {
            activity?.SetStatus(ActivityStatusCode.Error, "canceled");
            throw;
        }
        catch (Exception exception)
        {
            activity?.SetTag("exception.type", exception.GetType().FullName);
            activity?.SetTag("exception.message", exception.Message);
            activity?.SetStatus(ActivityStatusCode.Error, exception.Message);

            _logger.LogError(
                exception,
                "Impossibile ottenere una risposta dal modello upstream {Endpoint} usando il modello {Model}. VerrÃ  restituito il messaggio standard di errore.",
                _options.UpstreamEndpoint,
                model);

            return BuildCompletionResponse(model, SanitizeAssistantContent(_options.StandardErrorMessage));
        }
    }

    public async Task WriteStreamingCompletionAsync(
        OpenAIChatCompletionRequest request,
        HttpResponse response,
        CancellationToken cancellationToken)
    {
        using var activity = ActivitySource.StartActivity("llm.stream_completion", ActivityKind.Internal);
        activity?.SetTag("llm.model", request.Model ?? _options.DefaultModel);
        activity?.SetTag("llm.stream", true);
        activity?.SetTag("llm.message_count", request.Messages.Count);
        activity?.SetTag("llm.tool_choice.kind", request.ToolChoice.ValueKind.ToString());

        var model = ResolveModel(request.Model);
        var kernel = BuildKernel(model);
        var chatHistory = BuildChatHistory(request);
        var chatService = kernel.GetRequiredService<IChatCompletionService>();
        var settings = BuildExecutionSettings(request, kernel);
        var completionId = $"chatcmpl-{Guid.NewGuid():N}";
        var created = DateTimeOffset.UtcNow.ToUnixTimeSeconds();

        response.StatusCode = StatusCodes.Status200OK;
        response.ContentType = "text/event-stream";
        response.Headers.CacheControl = "no-cache";
        response.Headers["X-Accel-Buffering"] = "no";

        var roleChunk = new OpenAIChatCompletionChunk
        {
            Id = completionId,
            Created = created,
            Model = model,
            Choices =
            [
                new OpenAIChatChunkChoice
                {
                    Index = 0,
                    Delta = new OpenAIChatDelta
                    {
                        Role = "assistant"
                    }
                }
            ]
        };

        await WriteSseAsync(response, roleChunk, cancellationToken);

        var rawContentBuilder = new StringBuilder();
        var emittedContent = string.Empty;

        var streamSucceeded = false;
        try
        {
            _logger.LogInformation(
                "Elaborazione chat completion streaming con modello {Model}. Messaggi={MessageCount}",
                model,
                request.Messages.Count);

            await foreach (var update in chatService.GetStreamingChatMessageContentsAsync(
                               chatHistory,
                               settings,
                               kernel,
                               cancellationToken))
            {
                var contentPiece = update.Content ?? update.ToString();
                if (string.IsNullOrEmpty(contentPiece))
                {
                    continue;
                }

                rawContentBuilder.Append(contentPiece);
                var sanitizedContent = SanitizeAssistantContent(rawContentBuilder.ToString());
                var delta = ExtractStreamingDelta(emittedContent, sanitizedContent);
                if (string.IsNullOrEmpty(delta))
                {
                    continue;
                }

                emittedContent += delta;

                var contentChunk = new OpenAIChatCompletionChunk
                {
                    Id = completionId,
                    Created = created,
                    Model = model,
                    Choices =
                    [
                        new OpenAIChatChunkChoice
                        {
                            Index = 0,
                            Delta = new OpenAIChatDelta
                            {
                                Content = delta
                            }
                        }
                    ]
                };

                await WriteSseAsync(response, contentChunk, cancellationToken);
            }

            streamSucceeded = true;
        }
        catch (OperationCanceledException)
        {
            activity?.SetStatus(ActivityStatusCode.Error, "canceled");
            throw;
        }
        catch (Exception exception)
        {
            activity?.SetTag("exception.type", exception.GetType().FullName);
            activity?.SetTag("exception.message", exception.Message);
            activity?.SetStatus(ActivityStatusCode.Error, exception.Message);

            _logger.LogError(
                exception,
                "Errore durante lo streaming dal modello upstream {Endpoint} usando il modello {Model}.",
                _options.UpstreamEndpoint,
                model);

            if (string.IsNullOrEmpty(emittedContent))
            {
                var fallbackContent = SanitizeAssistantContent(_options.StandardErrorMessage);
                if (!string.IsNullOrWhiteSpace(fallbackContent))
                {
                    var errorChunk = new OpenAIChatCompletionChunk
                    {
                        Id = completionId,
                        Created = created,
                        Model = model,
                        Choices =
                        [
                            new OpenAIChatChunkChoice
                            {
                                Index = 0,
                                Delta = new OpenAIChatDelta
                                {
                                    Content = fallbackContent
                                }
                            }
                        ]
                    };

                    await WriteSseAsync(response, errorChunk, cancellationToken);
                }
            }
        }

        if (streamSucceeded)
        {
            activity?.SetTag("llm.response.emitted_chars", emittedContent.Length);
            activity?.SetStatus(ActivityStatusCode.Ok);
        }

        var stopChunk = new OpenAIChatCompletionChunk
        {
            Id = completionId,
            Created = created,
            Model = model,
            Choices =
            [
                new OpenAIChatChunkChoice
                {
                    Index = 0,
                    Delta = new OpenAIChatDelta(),
                    FinishReason = "stop"
                }
            ]
        };

        await WriteSseAsync(response, stopChunk, cancellationToken);
        await response.WriteAsync("data: [DONE]\n\n", cancellationToken);
        await response.Body.FlushAsync(cancellationToken);
    }

    private Kernel BuildKernel(string model)
    {
        using var activity = ActivitySource.StartActivity("llm.build_kernel", ActivityKind.Internal);
        activity?.SetTag("llm.model", model);
        activity?.SetTag("llm.upstream_endpoint", _options.UpstreamEndpoint);

        var builder = Kernel.CreateBuilder();
        builder.AddOpenAIChatCompletion(
            modelId: model,
            endpoint: new Uri(_options.UpstreamEndpoint),
            apiKey: _options.ApiKey,
            orgId: _options.OrganizationId);

        var kernel = builder.Build();
        kernel.FunctionInvocationFilters.Add(new ToolInvocationLoggingFilter(_logger));
        kernel.ImportPluginFromType<TicketApertiPlugin>();
        activity?.SetStatus(ActivityStatusCode.Ok);

        return kernel;
    }

    private ChatHistory BuildChatHistory(OpenAIChatCompletionRequest request)
    {
        using var activity = ActivitySource.StartActivity("llm.build_chat_history", ActivityKind.Internal);
        activity?.SetTag("llm.message_count", request.Messages.Count);
        activity?.SetTag("llm.tool_choice.kind", request.ToolChoice.ValueKind.ToString());

        var chatHistory = new ChatHistory();

        if (!string.IsNullOrWhiteSpace(_options.ServerInstruction))
        {
            chatHistory.AddSystemMessage(_options.ServerInstruction);
        }

        var lastUserMessage = request.Messages
            .LastOrDefault(message => string.Equals(message.Role, "user", StringComparison.OrdinalIgnoreCase))
            ?.ExtractTextContent();

        if (LooksLikeTicketRequest(lastUserMessage))
        {
            chatHistory.AddSystemMessage(
                "Per questa richiesta devi usare uno dei tool ticket prima di rispondere. " +
                "Se l'utente chiede quanti ticket aperti ha un cliente, usa conta_ticket_aperti_cliente. " +
                "Se l'utente chiede l'elenco dei ticket aperti di un cliente, usa recupera_ticket_aperti_cliente. " +
                "Se l'utente chiede il dettaglio di un ticket specifico, usa recupera_dettaglio_ticket. " +
                "Se manca il nome del cliente o l'ID ticket, chiedi solo il dato mancante.");
        }

        var toolChoiceInstruction = BuildToolChoiceInstruction(request.ToolChoice);
        if (!string.IsNullOrWhiteSpace(toolChoiceInstruction))
        {
            chatHistory.AddSystemMessage(toolChoiceInstruction);
        }

        foreach (var message in request.Messages)
        {
            var content = message.ExtractTextContent();
            switch (message.Role.Trim().ToLowerInvariant())
            {
                case "system":
                    chatHistory.AddSystemMessage(content);
                    break;
                case "developer":
                    chatHistory.AddSystemMessage(content);
                    break;
                case "user":
                    chatHistory.AddUserMessage(content);
                    break;
                case "assistant":
                    chatHistory.AddAssistantMessage(content);
                    break;
                case "tool":
                    chatHistory.AddMessage(AuthorRole.Tool, content);
                    break;
                default:
                    _logger.LogWarning("Ruolo non supportato {Role}; viene trattato come messaggio utente.", message.Role);
                    chatHistory.AddUserMessage(content);
                    break;
            }
        }

        activity?.SetStatus(ActivityStatusCode.Ok);

        return chatHistory;
    }

    private static OpenAIPromptExecutionSettings BuildExecutionSettings(
        OpenAIChatCompletionRequest request,
        Kernel kernel)
    {
        _ = kernel;

        var settings = new OpenAIPromptExecutionSettings
        {
            Temperature = request.Temperature,
            TopP = request.TopP,
            MaxTokens = request.MaxTokens,
            User = request.User
        };

        var lastUserMessage = request.Messages
            .LastOrDefault(message => string.Equals(message.Role, "user", StringComparison.OrdinalIgnoreCase))
            ?.ExtractTextContent();

        if (LooksLikeTicketRequest(lastUserMessage))
        {
            settings.FunctionChoiceBehavior = FunctionChoiceBehavior.Required(
                functions: null,
                autoInvoke: true);
            return settings;
        }

        if (IsToolChoiceRequired(request.ToolChoice))
        {
            settings.FunctionChoiceBehavior = FunctionChoiceBehavior.Required(
                functions: null,
                autoInvoke: true);
            return settings;
        }

        if (IsToolChoiceNone(request.ToolChoice))
        {
            settings.FunctionChoiceBehavior = FunctionChoiceBehavior.None();
            return settings;
        }

        settings.FunctionChoiceBehavior = FunctionChoiceBehavior.Auto(
            functions: null,
            autoInvoke: true);

        return settings;
    }

    private string ResolveModel(string? requestedModel)
    {
        return string.IsNullOrWhiteSpace(requestedModel)
            ? _options.DefaultModel
            : requestedModel;
    }

    private static OpenAIChatCompletionResponse BuildCompletionResponse(string model, string content)
    {
        return new OpenAIChatCompletionResponse
        {
            Id = $"chatcmpl-{Guid.NewGuid():N}",
            Created = DateTimeOffset.UtcNow.ToUnixTimeSeconds(),
            Model = model,
            Choices =
            [
                new OpenAIChatChoice
                {
                    Index = 0,
                    Message = new OpenAIAssistantMessage
                    {
                        Content = content
                    }
                }
            ]
        };
    }

    private static string SanitizeAssistantContent(string content)
    {
        if (string.IsNullOrWhiteSpace(content))
        {
            return string.Empty;
        }

        var sanitized = ThinkBlockRegex.Replace(content, string.Empty);
        sanitized = sanitized.Replace("<think>", string.Empty, StringComparison.OrdinalIgnoreCase);
        sanitized = sanitized.Replace("</think>", string.Empty, StringComparison.OrdinalIgnoreCase);

        return sanitized.Trim();
    }


    private static string ExtractStreamingDelta(string emittedContent, string sanitizedContent)
    {
        if (string.IsNullOrEmpty(sanitizedContent))
        {
            return string.Empty;
        }

        if (string.IsNullOrEmpty(emittedContent))
        {
            return sanitizedContent;
        }

        if (sanitizedContent.StartsWith(emittedContent, StringComparison.Ordinal))
        {
            return sanitizedContent[emittedContent.Length..];
        }

        return sanitizedContent;
    }

    private static string FormatArguments(KernelArguments arguments)
    {
        var values = arguments.ToDictionary(
            pair => pair.Key,
            pair => pair.Value?.ToString());

        return JsonSerializer.Serialize(values);
    }

    private static string FormatResult(object? result)
    {
        if (result is null)
        {
            return "null";
        }

        return result switch
        {
            string text => text,
            _ => JsonSerializer.Serialize(result)
        };
    }

    private static bool LooksLikeTicketRequest(string? message)
    {
        return !string.IsNullOrWhiteSpace(message) &&
               message.Contains("ticket", StringComparison.OrdinalIgnoreCase);
    }

    private static string? BuildToolChoiceInstruction(JsonElement toolChoice)
    {
        if (toolChoice.ValueKind is JsonValueKind.Undefined or JsonValueKind.Null)
        {
            return null;
        }

        if (toolChoice.ValueKind == JsonValueKind.String)
        {
            var choice = toolChoice.GetString();
            if (string.Equals(choice, "required", StringComparison.OrdinalIgnoreCase))
            {
                return "Devi eseguire almeno un tool disponibile prima di fornire la risposta finale.";
            }

            return null;
        }

        if (toolChoice.ValueKind != JsonValueKind.Object)
        {
            return null;
        }

        if (toolChoice.TryGetProperty("type", out var typeProperty) &&
            typeProperty.ValueKind == JsonValueKind.String &&
            string.Equals(typeProperty.GetString(), "function", StringComparison.OrdinalIgnoreCase) &&
            toolChoice.TryGetProperty("function", out var functionProperty) &&
            functionProperty.ValueKind == JsonValueKind.Object &&
            functionProperty.TryGetProperty("name", out var functionNameProperty) &&
            functionNameProperty.ValueKind == JsonValueKind.String)
        {
            var functionName = functionNameProperty.GetString();
            if (!string.IsNullOrWhiteSpace(functionName))
            {
                return $"Devi usare il tool {functionName} prima di fornire la risposta finale.";
            }
        }

        return null;
    }

    private static bool IsToolChoiceRequired(JsonElement toolChoice)
    {
        if (toolChoice.ValueKind is JsonValueKind.Undefined or JsonValueKind.Null)
        {
            return false;
        }

        if (toolChoice.ValueKind == JsonValueKind.String)
        {
            return string.Equals(toolChoice.GetString(), "required", StringComparison.OrdinalIgnoreCase);
        }

        if (toolChoice.ValueKind != JsonValueKind.Object)
        {
            return false;
        }

        return toolChoice.TryGetProperty("type", out var typeProperty) &&
               typeProperty.ValueKind == JsonValueKind.String &&
               string.Equals(typeProperty.GetString(), "function", StringComparison.OrdinalIgnoreCase);
    }

    private static bool IsToolChoiceNone(JsonElement toolChoice)
    {
        return toolChoice.ValueKind == JsonValueKind.String &&
               string.Equals(toolChoice.GetString(), "none", StringComparison.OrdinalIgnoreCase);
    }

    private sealed class ToolInvocationLoggingFilter(ILogger logger) : IFunctionInvocationFilter
    {
        public async Task OnFunctionInvocationAsync(
            FunctionInvocationContext context,
            Func<FunctionInvocationContext, Task> next)
        {
            logger.LogInformation(
                "Invocazione tool {Plugin}.{Function} con argomenti {Arguments}",
                context.Function.PluginName,
                context.Function.Name,
                FormatArguments(context.Arguments));

            await next(context);

            logger.LogInformation(
                "Tool completato {Plugin}.{Function} con risultato {Result}",
                context.Function.PluginName,
                context.Function.Name,
                FormatResult(context.Result.ToString()));
        }
    }

    private static async Task WriteSseAsync<T>(
        HttpResponse response,
        T payload,
        CancellationToken cancellationToken)
    {
        var json = JsonSerializer.Serialize(payload, SseSerializerOptions);
        await response.WriteAsync($"data: {json}\n\n", cancellationToken);
        await response.Body.FlushAsync(cancellationToken);
    }
}
