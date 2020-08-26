def main():
    import re
    import sys
    import json

    from nltk import casual_tokenize

    from src.preprocessor.preprocess import CODE_TOKENISATION_REGEX
    from src.tagger.config import Configuration
    from src.evaluate import restore_model
    from xmlrpc.server import SimpleXMLRPCServer
    from xmlrpc.server import SimpleXMLRPCRequestHandler

    marked_code_tokens_regex = re.compile(r'<tt>(.+)</tt>', flags=re.MULTILINE)

    # Restrict to a particular path.
    class RequestHandler(SimpleXMLRPCRequestHandler):
        rpc_paths = ('/RPC2',)

    # Create server
    with SimpleXMLRPCServer(("localhost", 8000), requestHandler=RequestHandler, allow_none=True) as server:
        # create instance of config
        config = Configuration()
        config.dir_model = sys.argv[1]
        model = restore_model(config)
        print('Loaded model from %s' % config.dir_model)
        casual = False

        # Define helper function
        def process_sent(sentence):
            if casual:
                words_raw = casual_tokenize(sentence.strip())
            else:
                words_raw = [l.strip()
                             for l in re.findall(CODE_TOKENISATION_REGEX,
                                                 sentence.strip())
                             if len(l.strip()) > 0]

            predictions = model.predict([marked_code_tokens_regex.sub(r"\1", w) for w in words_raw])
            if config.multilang:
                out = list()
                for w, predictions in zip(words_raw, zip(*predictions)):
                    current_out = {'word': str(w), 'language': str(int(predictions[-1]))}
                    for lid, tag in enumerate(predictions[:-1]):
                        current_out['tag_%d' % lid] = tag
            else:
                if isinstance(predictions, tuple):
                    out = [{'word': str(w), 'tag': str(t), 'language': str(int(lid))} for w, (t, lid) in
                           zip(words_raw, zip(*predictions))]
                else:
                    out = [{'word': str(w), 'tag': str(t)} for w, t in zip(words_raw, predictions)]
            print(json.dumps(out))
            return out

        server.register_introspection_functions()

        # Register an instance; all the methods of the instance are
        # published as XML-RPC methods
        class PredictionFunctions:
            def predict_sent(self, sentence):
                return process_sent(sentence)

            def is_multilang(self,):
                return {'multilang': str(config.multilang)}

        server.register_instance(PredictionFunctions())
        print('Loading done, entering serve loop')
        # Run the server's main loop
        server.serve_forever()


if __name__ == "__main__":
    main()
