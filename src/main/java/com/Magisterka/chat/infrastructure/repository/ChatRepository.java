package com.Magisterka.chat.infrastructure.repository;

import com.Magisterka.chat.domain.contract.DTO.*;
import lombok.RequiredArgsConstructor;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Repository;
import org.springframework.web.client.RestTemplate;

import java.util.List;
import java.util.Objects;

@Repository
@RequiredArgsConstructor
public class ChatRepository {

    private static final String GENERATE_ENDPOINT = "/api/generate";

    private final RestTemplate restTemplate;

    @Value("${ollama.base-url}")
    private String ollamaBaseUrl;

    @Value("${ollama.model}")
    private String ollamaModel;

    public ChatAnalyseResponse sendToAnalyse(ChatAnalyseRequest request) {
        OllamaGenerateRequest ollamaRequest = OllamaGenerateRequest.builder()
                .model(ollamaModel)
                .prompt(buildPrompt(request))
                .stream(false)
                .build();

        OllamaGenerateResponse response = restTemplate.postForObject(
                getOllamaGenerateUrl(),
                ollamaRequest,
                OllamaGenerateResponse.class
        );

        if (response == null || response.getResponse() == null) {
            throw new IllegalStateException("Empty response from analysis service");
        }

        return ChatAnalyseResponse.builder()
                .answer(response.getResponse().trim())
                .model(Objects.requireNonNullElse(response.getModel(), ollamaModel))
                .build();
    }

    public ChatHistoryResponse getHistory(ChatHistoryRequest request) {
        return ChatHistoryResponse.builder()
                .matchId(request.getMatchId())
                .messages(List.of())
                .build();
    }

    private String buildPrompt(ChatAnalyseRequest request) {
        StringBuilder prompt = new StringBuilder();
        prompt.append("You are a CS2 match analysis assistant. ");
        prompt.append("Answer with concise, practical match analysis. ");
        prompt.append("Focus on teams, maps, form, veto, risks, and prediction when relevant.");
        prompt.append(System.lineSeparator()).append(System.lineSeparator());

        appendHistory(prompt, request.getHistory());

        prompt.append("User: ").append(request.getMessage().trim()).append(System.lineSeparator());
        prompt.append("Assistant:");

        return prompt.toString();
    }

    private void appendHistory(StringBuilder prompt, List<ChatMessage> history) {
        if (history == null || history.isEmpty()) {
            return;
        }

        prompt.append("Conversation so far:").append(System.lineSeparator());
        history.stream()
                .filter(Objects::nonNull)
                .filter(message -> message.getContent() != null && !message.getContent().isBlank())
                .forEach(message -> {
                    String role = "assistant".equalsIgnoreCase(message.getRole()) ? "Assistant" : "User";
                    prompt.append(role)
                            .append(": ")
                            .append(message.getContent().trim())
                            .append(System.lineSeparator());
                });
        prompt.append(System.lineSeparator());
    }

    private String getOllamaGenerateUrl() {
        return ollamaBaseUrl.trim().replaceAll("/+$", "") + GENERATE_ENDPOINT;
    }
}
