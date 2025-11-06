Human Follower
--------------

This example uses a servo motor to point at humans detected by the "People detection" model.

You should first use the Sensecraft AI web site to load this model into your Grove Vision AI V2 board.

Steps:

    1. Flash a person or face detection model from Sensecraft AI to the Grove Vision AI V2
       using its USB-C connector.
       I used this model: https://sensecraft.seeed.cc/ai/view-model/60086-person-detection-swift-yolo

    2. Copy ``human_follower.mpy`` to your ``CIRCUITPY/`` drive.

    3. Copy ``grove_vision_ai_v2.mpy`` to ``CIRCUITPY/lib/``

    4. Create ``CIRCUITPY/code.py`` with the following content:

.. code-block:: python

   import humanfollower


.. literalinclude:: ../examples/human_follower.py
    :caption: examples/human_follower.py
    :linenos:
