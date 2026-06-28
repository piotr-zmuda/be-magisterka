package com.Magisterka.chat.domain.contract;

import com.Magisterka.chat.domain.contract.DTO.ChatAnalyseRequest;
import com.Magisterka.chat.domain.contract.DTO.ChatAnalyseResponse;
import com.Magisterka.chat.domain.contract.DTO.ChatHistoryRequest;
import com.Magisterka.chat.domain.contract.DTO.ChatHistoryResponse;
import io.swagger.v3.oas.annotations.Operation;
import io.swagger.v3.oas.annotations.media.Content;
import io.swagger.v3.oas.annotations.media.Schema;
import io.swagger.v3.oas.annotations.responses.ApiResponse;
import io.swagger.v3.oas.annotations.responses.ApiResponses;
import org.springframework.http.HttpStatus;
import org.springframework.web.ErrorResponse;
import org.springframework.web.bind.annotation.*;

import static com.Magisterka.shared.ApplicationMappings.CHAT;
import static com.Magisterka.shared.ApplicationMappings.JSON;

@RequestMapping(CHAT)
@CrossOrigin(origins = "*")
public interface ChatResources {

    String SEND_TO_ANALYSE = "/send-to-analyse";
    String GET_HISTORY = "/get-history";

    @PostMapping(value = SEND_TO_ANALYSE, consumes = JSON, produces = JSON)
    @ResponseStatus(HttpStatus.OK)
    @Operation(summary = "Send chat message to analyse")
    @ApiResponses(value = {
            @ApiResponse(
                    responseCode = "200",
                    description = "Analysis generated successfully",
                    content = @Content(
                            mediaType = JSON,
                            schema = @Schema(implementation = ChatAnalyseResponse.class)
                    )
            ),
            @ApiResponse(
                    responseCode = "400",
                    description = "Invalid request",
                    content = @Content(mediaType = JSON, schema = @Schema(implementation = ErrorResponse.class))
            ),
            @ApiResponse(
                    responseCode = "500",
                    description = "Internal server error",
                    content = @Content(mediaType = JSON, schema = @Schema(implementation = ErrorResponse.class))
            ),
    })
    ChatAnalyseResponse sendToAnalyse(@RequestBody ChatAnalyseRequest request);

    @PostMapping(value = GET_HISTORY, consumes = JSON, produces = JSON)
    @ResponseStatus(HttpStatus.OK)
    @Operation(summary = "Get chat history for selected match")
    @ApiResponses(value = {
            @ApiResponse(
                    responseCode = "200",
                    description = "Chat history fetched successfully",
                    content = @Content(
                            mediaType = JSON,
                            schema = @Schema(implementation = ChatHistoryResponse.class)
                    )
            ),
            @ApiResponse(
                    responseCode = "400",
                    description = "Invalid request",
                    content = @Content(mediaType = JSON, schema = @Schema(implementation = ErrorResponse.class))
            ),
            @ApiResponse(
                    responseCode = "500",
                    description = "Internal server error",
                    content = @Content(mediaType = JSON, schema = @Schema(implementation = ErrorResponse.class))
            ),
    })
    ChatHistoryResponse getHistory(@RequestBody ChatHistoryRequest request);
}
