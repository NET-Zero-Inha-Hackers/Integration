const WEBSOCKET_URL = "wss://optimo-chat-ssl.ptg.kr";

function createWebSocket(
  uri,
  prompt,
  jwtToken,
  modelNameHandler,
  modelResponseHandler, //
  metadataHandler
) {
  const socket = new WebSocket(`${WEBSOCKET_URL}${uri}`);
  socket.onmessage = (event) => {
    const data = JSON.parse(event.data);
    if (data.message == "connected") {
      //   console.log("WebSocket 연결됨:", data);
    } else if (data.message == "model selected") {
      //   console.log("모델 선택됨:", data);
      modelNameHandler(data.model);
    } else if (
      typeof data.id === "number" &&
      data.text !== undefined &&
      data.text !== null
    ) {
      //   console.log("모델 출력 수신:", data.id);
      modelResponseHandler(data.id, data.text);
    } else if (data.end === "end") {
      //   console.log("모델 출력 종료:", data.id);
      if (metadataHandler !== null) {
        metadataHandler(data.id, data.title, data.description);
      }
    }
  };
  socket.onopen = () => {
    console.log("WebSocket 연결 성공");
    if (prompt) {
      socket.send(JSON.stringify({ prompt, jwtToken }));
    }
  };
  return socket;
}

export function newChatWebSocket(
  prompt,
  jwtToken,
  modelNameHandler,
  modelResponseHandler,
  metadataHandler
) {
  const uri = "/chat/new";
  return createWebSocket(
    uri,
    prompt,
    jwtToken,
    modelNameHandler,
    modelResponseHandler,
    metadataHandler
  );
}

export function continueChatWebSocket(
  chatId,
  prompt,
  jwtToken,
  modelNameHandler,
  modelResponseHandler
) {
  const uri = `/chat/continue/${chatId}`;
  return createWebSocket(
    uri,
    prompt,
    jwtToken,
    modelNameHandler,
    modelResponseHandler,
    null // 메타데이터 핸들러는 필요하지 않음(이미 프론트는 값을 가지고 있답니다)
  );
}
